"""Dynamic EPLB controller backed by NewModelLoader checkpoint streaming."""

import logging
import random
import threading
from collections import deque
from enum import Enum
from queue import Queue
from typing import Any, Deque, Dict, Sequence

import torch

from rtp_llm.eplb.eplb import rebalance_experts
from rtp_llm.models_py.expert_placement import ExpertPlacement
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


class _HistoryStats:
    def __init__(self, window_size: int, shape: Sequence[int]):
        if isinstance(window_size, bool) or not isinstance(window_size, int):
            raise TypeError("eplb_stats_window_size must be an integer")
        if window_size <= 0:
            raise ValueError("eplb_stats_window_size must be positive")
        self._window_size = window_size
        self._values: Deque[torch.Tensor] = deque()
        self._total = torch.zeros(tuple(shape), dtype=torch.int64)

    def add(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device="cpu", dtype=torch.int64)
        if value.shape != self._total.shape:
            raise ValueError(
                f"EPLB logical stats shape changed from {tuple(self._total.shape)} "
                f"to {tuple(value.shape)}"
            )
        if len(self._values) == self._window_size:
            self._total.sub_(self._values.popleft())
        snapshot = value.clone()
        self._values.append(snapshot)
        self._total.add_(snapshot)
        return self._total


class _SelectLayerMethod(Enum):
    ROUND = "round"
    RANDOM = "random"
    MOST_UNBALANCED_LAYER = "most_unbalanced_layer"
    MIX = "mix"


class NewLoaderExpertBalancer:
    """Implements the Python callback protocol consumed by C++ ExpertBalancer."""

    def supports_python_weight_install(self) -> bool:
        """Tell C++ that NewLoader installs CPU-staged tensors into Python weights."""
        return True

    def __init__(self, loader: Any, model: torch.nn.Module, model_config: Any):
        self._loader = loader
        self._model = model
        self._model_config = model_config
        self._lock = threading.Lock()

        experts: Dict[int, BaseMoEExperts] = {}
        for module in model.modules():
            if not isinstance(module, BaseMoEExperts):
                continue
            if module.layer_idx in experts:
                raise RuntimeError(
                    f"Multiple routed-expert modules found for layer {module.layer_idx}"
                )
            experts[module.layer_idx] = module
        if not experts:
            raise ValueError("Dynamic EPLB requires at least one NewLoader MoE layer")
        self._experts = experts
        self.moe_layer_index = tuple(sorted(experts))

        first = experts[self.moe_layer_index[0]]
        self.num_layers = int(getattr(model_config, "num_layers", len(experts)))
        self.num_experts = first.num_experts
        self.num_replicas = first.physical_num_experts
        self.num_gpu = first.ep_size
        self.num_groups = int(getattr(model_config, "moe_n_group", 1))
        if self.num_groups <= 0:
            raise ValueError("model_config.moe_n_group must be positive")

        parallelism = loader.load_config.parallelism_config
        world_size = int(getattr(parallelism, "world_size", self.num_gpu))
        local_world_size = int(getattr(parallelism, "local_world_size", world_size))
        if local_world_size <= 0 or world_size % local_world_size:
            raise ValueError(
                "parallelism world_size must be divisible by local_world_size"
            )
        self.num_nodes = max(1, world_size // local_world_size)
        if self.num_gpu % self.num_nodes:
            self.num_nodes = 1

        eplb_config = model_config.eplb_config
        self.select_layer_method = _SelectLayerMethod(eplb_config.balance_method)
        self.force_repack = int(eplb_config.eplb_force_repack) == 1
        self._round_robin: Queue[int] = Queue()
        self._update_count = 0
        self._history = _HistoryStats(
            int(eplb_config.eplb_stats_window_size),
            (self.num_layers, self.num_experts),
        )
        self.phy2log = [list(range(self.num_replicas)) for _ in range(self.num_layers)]
        for layer_id, module in experts.items():
            placement = module._expert_placement
            if (
                placement.logical_expert_count != self.num_experts
                or placement.physical_expert_count != self.num_replicas
                or placement.ep_size != self.num_gpu
            ):
                raise ValueError("All EPLB MoE layers must share one expert topology")
            self.phy2log[layer_id] = list(placement.physical_to_logical)

    def _select_layer(self, gpu_loads: torch.Tensor) -> int:
        method = self.select_layer_method
        if method == _SelectLayerMethod.ROUND:
            return self._next_round_robin()
        if method == _SelectLayerMethod.RANDOM:
            return random.choice(self.moe_layer_index)
        if method == _SelectLayerMethod.MOST_UNBALANCED_LAYER:
            return self._most_unbalanced(gpu_loads)
        if method == _SelectLayerMethod.MIX:
            return (
                self._next_round_robin()
                if self._update_count % 2 == 0
                else self._most_unbalanced(gpu_loads)
            )
        raise RuntimeError(f"Unsupported EPLB balance method {method}")

    def _next_round_robin(self) -> int:
        if self._round_robin.empty():
            for layer_id in self.moe_layer_index:
                self._round_robin.put(layer_id)
        layer_id = self._round_robin.get()
        self._round_robin.put(layer_id)
        return layer_id

    def _most_unbalanced(self, gpu_loads: torch.Tensor) -> int:
        if gpu_loads.dim() != 2 or gpu_loads.shape[0] != self.num_layers:
            raise ValueError(
                f"EPLB gpu_loads must have shape [{self.num_layers}, ep_size]"
            )
        maxima = gpu_loads.max(dim=1).values
        return max(self.moe_layer_index, key=lambda layer_id: int(maxima[layer_id]))

    @torch.inference_mode()
    def create_balance_plan(self, log_stats: torch.Tensor, gpu_loads: torch.Tensor):
        with self._lock:
            self._update_count += 1
            history = self._history.add(log_stats)
            layer_id = self._select_layer(gpu_loads.to(device="cpu"))
            phy2log, _, _ = rebalance_experts(
                history[layer_id : layer_id + 1],
                self.num_replicas,
                self.num_groups,
                self.num_nodes,
                self.num_gpu,
                self.force_repack,
            )
            layer_phy2log = phy2log[0].to(dtype=torch.int32, device="cpu")
            self.phy2log[layer_id] = layer_phy2log.tolist()
            module = self._experts[layer_id]
            placement = ExpertPlacement(
                logical_expert_count=self.num_experts,
                ep_size=module.ep_size,
                ep_rank=module.ep_rank,
                physical_to_logical=tuple(layer_phy2log.tolist()),
            )
            # Treat phy2log as the one placement source of truth.  Rebuild the
            # inverse tensors from it so the loader and runtime cannot observe
            # mutually inconsistent routing metadata after repeated updates.
            logic_expert_cnt, log2phy = module._routing_tensors(placement)
        logger.info(
            "[NewLoader EPLB] plan layer=%d phy2log=%s",
            layer_id,
            layer_phy2log.tolist(),
        )
        return (
            torch.tensor([layer_id], dtype=torch.int32).contiguous(),
            logic_expert_cnt.contiguous(),
            log2phy.contiguous(),
            layer_phy2log.contiguous(),
        )

    @torch.inference_mode()
    def load_moe_weight(
        self,
        layer_id_tensor: torch.Tensor,
        ep_rank: int,
        ep_size: int,
        phy2log: torch.Tensor,
    ):
        layer_id = int(layer_id_tensor.item())
        if layer_id not in self._experts:
            raise ValueError(f"Layer {layer_id} is not a NewLoader MoE layer")
        planned_phy2log = [int(value) for value in phy2log.to(device="cpu").tolist()]
        with self._lock:
            # create_balance_plan only runs on rank 0.  Record the broadcast
            # plan during each rank's load callback so controller state stays
            # coherent everywhere, and can be restored if any rank fails.
            self.phy2log[layer_id] = planned_phy2log
        weights = self._loader.load_moe_layer_weights(
            self._model,
            layer_id=layer_id,
            physical_to_logical=planned_phy2log,
            ep_rank=ep_rank,
            ep_size=ep_size,
        )
        # This callback runs inside the C++ loading worker.  A mismatch must
        # therefore become a load failure and participate in the all-rank
        # rollback before any live tensor is touched.
        self._experts[layer_id].validate_eplb_reload_weights(weights)
        return (
            layer_id,
            weights[W.moe_w1],
            weights[W.moe_w2],
            weights.get(W.moe_s1, torch.empty(0, dtype=torch.float32)),
            weights.get(W.moe_s2, torch.empty(0, dtype=torch.float32)),
        )

    def rollback_balance_plan(self, layer_id: int) -> None:
        """Restore controller state from the last plan installed in the model."""
        if layer_id not in self._experts:
            raise ValueError(f"Layer {layer_id} is not a NewLoader MoE layer")
        self._experts[layer_id].finalize_eplb_install(commit=False)
        installed = list(self._experts[layer_id]._expert_placement.physical_to_logical)
        with self._lock:
            self.phy2log[layer_id] = installed
        logger.warning(
            "[NewLoader EPLB] rolled back uninstalled plan for layer=%d", layer_id
        )

    def commit_balance_plan(self, layer_id: int) -> None:
        if layer_id not in self._experts:
            raise ValueError(f"Layer {layer_id} is not a NewLoader MoE layer")
        self._experts[layer_id].finalize_eplb_install(commit=True)
