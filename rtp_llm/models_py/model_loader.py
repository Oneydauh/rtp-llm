import inspect
import logging
import os
import platform
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import rtp_llm.models_py.weight_mapper as weight_mapper
import torch
import torch.nn as nn
from rtp_llm.models_py.expert_placement import (
    ExpertPlacement,
    expert_placement_for_layer,
)
from rtp_llm.models_py.module_base import (
    CheckpointLoadTarget,
    RtpModule,
    collect_loaded_tensor_ids,
)
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor, ReadSpec
from rtp_llm.models_py.registry import get_model_class, list_models

logger = logging.getLogger(__name__)
_CUSTOM_WEIGHT_PREFIX = "__custom__."

_EXPERT_ID_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_EXPERT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_STACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:.*\.)?experts)\."
    r"(?P<projection>gate_up_proj|down_proj)"
    r"(?P<suffix>(?:\..+)?)$"
)
_FASTSAFETENSORS_REQUIRED_ARGUMENTS = frozenset(
    {
        "pg",
        "hf_weights_files",
        "use_tqdm_on_load",
        "device",
        "bbuf_size_kb",
        "use_shm",
        "nogds",
    }
)
_FASTSAFETENSORS_MIN_BBUF_SIZE_KB = 16 * 1024
_FASTSAFETENSORS_MAX_BBUF_SIZE_KB = 2 * 1024 * 1024
_CHECKPOINT_CATALOG_UNSET = object()
_CHECKPOINT_LOAD_PLAN_UNSET = object()


def _require_callable_name_filter(
    model_filter: Optional[Callable[[str], bool]], label: str
) -> Optional[Callable[[str], bool]]:
    if isinstance(model_filter, nn.Module):
        raise TypeError(f"{label} must be callable or None, not an nn.Module")
    if model_filter is not None and not callable(model_filter):
        raise TypeError(f"{label} must be callable or None")
    return model_filter


class _ExpertRangeFilter:
    """Select logical expert tensors needed by this EP rank's physical slots."""

    def __init__(
        self,
        num_experts: int,
        ep_size: int,
        ep_rank: int,
        model_config: Any = None,
        parallelism_config: Any = None,
        placement_overrides: Optional[Dict[int, ExpertPlacement]] = None,
    ):
        for name, value in (
            ("num_experts", num_experts),
            ("ep_size", ep_size),
            ("ep_rank", ep_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
        if num_experts <= 0 or num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible by "
                f"ep_size={ep_size}"
            )
        experts_per_rank = num_experts // ep_size
        self.start_expert = ep_rank * experts_per_rank
        self.end_expert = self.start_expert + experts_per_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts = num_experts
        self._model_config = model_config
        self._parallelism_config = parallelism_config
        self._placement_overrides = dict(placement_overrides or {})
        self._placement_cache: Dict[int, ExpertPlacement] = {}

    @staticmethod
    def _layer_index(name: str) -> int:
        match = _EXPERT_LAYER_RE.search(name)
        return int(match.group(1)) if match is not None else 0

    def _placement(self, name: str) -> Optional[ExpertPlacement]:
        if self._model_config is None:
            return None
        layer_index = self._layer_index(name)
        override = self._placement_overrides.get(layer_index)
        if override is not None:
            return override
        placement = self._placement_cache.get(layer_index)
        if placement is None:
            placement = expert_placement_for_layer(
                self._model_config,
                layer_index=layer_index,
                logical_expert_count=self.num_experts,
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                parallelism_config=self._parallelism_config,
            )
            self._placement_cache[layer_index] = placement
        return placement

    def expert_id(self, name: str) -> Optional[int]:
        match = _EXPERT_ID_RE.search(name)
        if match is None:
            return None
        expert_id = int(match.group(1))
        if not 0 <= expert_id < self.num_experts:
            raise ValueError(
                f"Checkpoint tensor {name!r} has expert id {expert_id} outside "
                f"[0, {self.num_experts})"
            )
        return expert_id

    def target_ep_ranks(self, name: str) -> Optional[Tuple[int, ...]]:
        expert_id = self.expert_id(name)
        if expert_id is None:
            return None
        placement = self._placement(name)
        if placement is not None:
            return placement.target_ep_ranks(expert_id)
        experts_per_rank = self.num_experts // self.ep_size
        return (expert_id // experts_per_rank,)

    def target_world_ranks(
        self, name: str, world_size: int
    ) -> Optional[Tuple[int, ...]]:
        target_ep_ranks = self.target_ep_ranks(name)
        if target_ep_ranks is None:
            return None
        return tuple(
            rank for rank in range(world_size) if rank % self.ep_size in target_ep_ranks
        )

    def should_load(self, name: str) -> bool:
        expert_id = self.expert_id(name)
        if expert_id is not None:
            placement = self._placement(name)
            if placement is not None:
                return bool(placement.local_indices(expert_id))
            return self.start_expert <= expert_id < self.end_expert
        if self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name):
            raise ValueError(
                "EP loading requires per-expert checkpoint keys; stacked all-expert "
                f"tensor {name!r} would materialize non-local experts"
            )
        return True

    def handles_safetensor(self, name: str) -> bool:
        return self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name) is not None

    def expand_safetensor(
        self, name: str, shape: Tuple[int, ...]
    ) -> List[Tuple[str, int]]:
        return [
            (read_spec.output_name, read_spec.slices[0])
            for read_spec in self.stacked_read_specs(name, shape)
            if read_spec.load
        ]

    def stacked_read_specs(
        self, name: str, shape: Tuple[int, ...]
    ) -> Tuple[ReadSpec, ...]:
        """Plan one stacked tensor as independently routed logical experts.

        FST first scatters the canonical contiguous expert ranges described by
        ``shard_rank``. It then sends only rows whose physical placement adds
        another target rank. Scratch applies the same specs as direct
        safetensors slices, so neither backend materializes the full all-expert
        tensor on every rank.
        """
        match = _STACKED_EXPERT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Tensor {name!r} is not a stacked expert tensor")
        if not shape or shape[0] != self.num_experts:
            raise ValueError(
                f"Stacked expert tensor {name!r} must contain "
                f"{self.num_experts} experts, got shape {shape}"
            )
        placement = self._placement(name)
        suffix = match.group("suffix") or ".weight"
        prefix = match.group("prefix")
        projection = match.group("projection")
        experts_per_rank = self.num_experts // self.ep_size
        specs = []
        for expert_id in range(self.num_experts):
            target_ranks = (
                placement.target_ep_ranks(expert_id)
                if placement is not None
                else (expert_id // experts_per_rank,)
            )
            specs.append(
                ReadSpec(
                    source_name=name,
                    output_name=f"{prefix}.{expert_id}.{projection}{suffix}",
                    source_shape=tuple(shape),
                    load=self.ep_rank in target_ranks,
                    slices=(expert_id,) + (slice(None),) * (len(shape) - 1),
                    partition="ep_stacked_expert",
                    shard_dim=0,
                    shard_count=self.ep_size,
                    shard_rank=expert_id // experts_per_rank,
                    target_ranks=target_ranks,
                )
            )
        return tuple(specs)


class _StagedFSTLoadCoordinator:
    """Quantize staged leaves as soon as their checkpoint inputs are complete."""

    def __init__(
        self,
        model: nn.Module,
        staged_modules: List[RtpModule],
        target_device: str,
    ):
        names = {module: name or "<root>" for name, module in model.named_modules()}
        self._module_names: Dict[nn.Module, str] = {
            module: names[module] for module in staged_modules
        }
        self._target_device = target_device
        self._prepared: Set[nn.Module] = set()
        self._finalized: Set[nn.Module] = set()
        self._peak_active = 0

    def handles(self, child: nn.Module) -> bool:
        return child in self._module_names

    def load_child(self, child: nn.Module, weights: Any) -> None:
        if child not in self._module_names:
            child.load_weights(weights)
            return
        if child in self._finalized:
            raise RuntimeError(
                "Checkpoint tensor arrived after staged postprocessing completed "
                f"for {self._module_names[child]} ({type(child).__name__})"
            )
        if child not in self._prepared:
            NewModelLoader._prepare_model_for_fastsafetensors(
                child, self._target_device
            )
            self._prepared.add(child)
            self._peak_active = max(
                self._peak_active, len(self._prepared - self._finalized)
            )
        child.load_weights(weights)
        if child.checkpoint_load_complete():
            child.process_weights_after_loading()
            self._finalized.add(child)

    def finish(self) -> None:
        unfinished = [
            f"{self._module_names[module]} ({type(module).__name__})"
            for module in self._module_names
            if module not in self._finalized
        ]
        if unfinished:
            raise RuntimeError(
                "FST staged postprocessing did not receive a complete checkpoint "
                f"for {unfinished[:10]}"
            )
        logger.info(
            "FST online quantization finalized %d leaves; peak concurrently "
            "staged leaves=%d",
            len(self._finalized),
            self._peak_active,
        )


def _validate_runtime_device(device: str, label: str) -> None:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {device!r}") from exc
    if parsed.type == "meta":
        raise ValueError(
            f"{label} cannot be meta; newloader requires materialized weights"
        )


class NewLoaderLoadMethod(str, Enum):
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


@dataclass(frozen=True)
class NewLoaderConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    load_method: NewLoaderLoadMethod = NewLoaderLoadMethod.AUTO
    quant_config: Any = None
    parallelism_config: Any = None
    moe_config: Any = None
    fmha_config: Any = None
    device_resource_config: Any = None
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    ffn_tp_size: Optional[int] = None
    ffn_tp_rank: Optional[int] = None
    lm_head_tp_size: Optional[int] = None
    lm_head_tp_rank: Optional[int] = None
    keep_mla_checkpoint_weights: bool = False
    custom_weight_mappings: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.load_method, str):
            try:
                object.__setattr__(
                    self,
                    "load_method",
                    NewLoaderLoadMethod(self.load_method.strip().lower()),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported newloader load method {self.load_method!r}"
                ) from exc
        elif not isinstance(self.load_method, NewLoaderLoadMethod):
            raise TypeError(
                f"load_method must be NewLoaderLoadMethod or str, got "
                f"{type(self.load_method).__name__}"
            )
        for name in ("tp_size", "tp_rank", "ep_size", "ep_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"Invalid TP partition: rank={self.tp_rank}, size={self.tp_size}"
            )
        for prefix in ("attn_tp", "ffn_tp", "lm_head_tp"):
            size_name = f"{prefix}_size"
            rank_name = f"{prefix}_rank"
            size = getattr(self, size_name)
            rank = getattr(self, rank_name)
            if size is None and rank is None:
                size, rank = self.tp_size, self.tp_rank
                object.__setattr__(self, size_name, size)
                object.__setattr__(self, rank_name, rank)
            elif size is None or rank is None:
                raise ValueError(
                    f"{size_name} and {rank_name} must be configured together"
                )
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError(f"{size_name} must be an integer")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"{rank_name} must be an integer")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(
                    f"Invalid {prefix} partition: rank={rank}, size={size}"
                )
            if prefix in ("attn_tp", "ffn_tp"):
                if size not in (1, self.tp_size):
                    raise ValueError(
                        f"{size_name}={size} must be either 1 or the physical "
                        f"tp_size={self.tp_size}; independent TP subgroups are "
                        "not supported by Group.TP collectives"
                    )
                expected_rank = 0 if size == 1 else self.tp_rank
                if rank != expected_rank:
                    raise ValueError(
                        f"{rank_name}={rank} does not match the supported "
                        f"{prefix} topology: expected rank={expected_rank} for "
                        f"size={size}"
                    )
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"Invalid EP partition: rank={self.ep_rank}, size={self.ep_size}"
            )
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("compute_dtype must be a torch.dtype")
        if not isinstance(self.keep_mla_checkpoint_weights, bool):
            raise TypeError("keep_mla_checkpoint_weights must be a bool")
        if not isinstance(self.custom_weight_mappings, tuple):
            raise TypeError("custom_weight_mappings must be a tuple")
        runtime_names = set()
        checkpoint_names = set()
        for mapping in self.custom_weight_mappings:
            if not isinstance(mapping, tuple) or len(mapping) != 2:
                raise TypeError(
                    "custom_weight_mappings entries must be "
                    "(runtime_name, checkpoint_name) tuples"
                )
            runtime_name, checkpoint_name = mapping
            if not isinstance(runtime_name, str) or not runtime_name:
                raise TypeError("custom runtime weight names must be non-empty strings")
            if not runtime_name.startswith(_CUSTOM_WEIGHT_PREFIX):
                raise ValueError(
                    f"custom runtime weight {runtime_name!r} must start with "
                    f"{_CUSTOM_WEIGHT_PREFIX!r}"
                )
            if not isinstance(checkpoint_name, str) or not checkpoint_name:
                raise TypeError(
                    "custom checkpoint weight names must be non-empty strings"
                )
            if runtime_name in runtime_names:
                raise ValueError(f"Duplicate custom runtime weight {runtime_name!r}")
            if checkpoint_name in checkpoint_names:
                raise ValueError(
                    f"Duplicate custom checkpoint weight {checkpoint_name!r}"
                )
            runtime_names.add(runtime_name)
            checkpoint_names.add(checkpoint_name)
        _validate_runtime_device(self.device, "device")
        if self.parallelism_config is not None:
            for prefix in ("tp", "ep"):
                configured_size = getattr(
                    self.parallelism_config, f"{prefix}_size", None
                )
                configured_rank = getattr(
                    self.parallelism_config, f"{prefix}_rank", None
                )
                if configured_size is None and configured_rank is None:
                    continue
                expected = (
                    (self.tp_size, self.tp_rank)
                    if prefix == "tp"
                    else (self.ep_size, self.ep_rank)
                )
                if (configured_size, configured_rank) != expected:
                    raise ValueError(
                        f"parallelism_config {prefix.upper()} partition does not "
                        f"match NewLoaderConfig: parallelism="
                        f"({configured_rank}, {configured_size}) loader="
                        f"({expected[1]}, {expected[0]})"
                    )
            topology_getters = {
                "attn_tp": ("get_attn_tp_size", "get_attn_tp_rank"),
                "ffn_tp": ("get_ffn_tp_size", "get_ffn_tp_rank"),
                "lm_head_tp": ("tp_size", "tp_rank"),
            }
            for prefix, (
                size_getter_name,
                rank_getter_name,
            ) in topology_getters.items():
                size_source = getattr(self.parallelism_config, size_getter_name, None)
                rank_source = getattr(self.parallelism_config, rank_getter_name, None)
                if callable(size_source) and callable(rank_source):
                    actual = (size_source(), rank_source())
                elif size_source is not None and rank_source is not None:
                    actual = (size_source, rank_source)
                else:
                    continue
                expected = (
                    getattr(self, f"{prefix}_size"),
                    getattr(self, f"{prefix}_rank"),
                )
                if actual != expected:
                    raise ValueError(
                        f"parallelism_config {prefix} partition does not match "
                        f"NewLoaderConfig: parallelism=({actual[1]}, {actual[0]}) "
                        f"loader=({expected[1]}, {expected[0]})"
                    )


class NewModelLoader:
    """Create a PyModel first, stream tensors into it, then run post-load hooks."""

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[NewLoaderConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        effective_config = NewLoaderConfig() if load_config is None else load_config
        if not isinstance(effective_config, NewLoaderConfig):
            raise TypeError(
                f"load_config must be NewLoaderConfig, got {type(effective_config).__name__}"
            )
        if device is not None:
            _validate_runtime_device(device, "device override")
            effective_config = replace(effective_config, device=device)
        self.model_config = model_config
        self.load_config = effective_config
        self.model_path = model_path
        self._ckpt_files = None
        self._safetensor_catalog_cache = _CHECKPOINT_CATALOG_UNSET
        self._selected_load_method: Optional[NewLoaderLoadMethod] = None

    @staticmethod
    def _moe_expert_for_layer(model: nn.Module, layer_id: int) -> RtpModule:
        matches = [
            module
            for module in model.modules()
            if bool(getattr(module, "_supports_newloader_eplb", False))
            and getattr(module, "layer_idx", None) == layer_id
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one NewLoader MoE expert module for layer {layer_id}, "
                f"found {len(matches)}"
            )
        return matches[0]

    @torch.inference_mode()
    def load_moe_layer_weights(
        self,
        model: nn.Module,
        *,
        layer_id: int,
        physical_to_logical: List[int],
        ep_rank: int,
        ep_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Reload one physical MoE layer for a dynamic EPLB placement."""
        if (ep_rank, ep_size) != (self.load_config.ep_rank, self.load_config.ep_size):
            raise ValueError(
                "C++ EPLB EP partition does not match NewLoader: "
                f"callback=({ep_rank}, {ep_size}) loader="
                f"({self.load_config.ep_rank}, {self.load_config.ep_size})"
            )
        live_experts = self._moe_expert_for_layer(model, layer_id)
        placement = ExpertPlacement(
            logical_expert_count=live_experts.num_experts,
            ep_size=ep_size,
            ep_rank=ep_rank,
            physical_to_logical=tuple(physical_to_logical),
        )
        if placement.physical_expert_count != live_experts.physical_num_experts:
            raise ValueError("Dynamic EPLB cannot change the physical expert count")

        # The callback is entered from a C++ std::thread.  CUDA's current
        # device is thread-local, so a newly-created loading thread otherwise
        # defaults to cuda:0 even when this process owns another local rank.
        # Pin the worker to the device of the live layer and use that explicit
        # device for every allocation/load below.  Using the rank-generic
        # ``load_config.device`` (usually just ``"cuda"``) can make two EP
        # ranks issue FST/NCCL operations from the same GPU and deadlock.
        live_device = live_experts.w13.device
        if live_device.type == "cuda":
            if live_device.index is None:
                raise RuntimeError(
                    "Dynamic EPLB live MoE weight must have an explicit CUDA device"
                )
            torch.cuda.set_device(live_device)
        reload_device = str(live_device)

        with torch.device("cpu"):
            reload_experts = live_experts.clone_empty_for_placement(placement)
        self._validate_runtime_backends(reload_experts, reload_device)

        model_filter = self._model_checkpoint_name_filter(model)
        prefix = live_experts.prefix + "."

        def target_layer_filter(name: str) -> bool:
            if model_filter is not None and not model_filter(name):
                return False
            mapped = self._mapped_checkpoint_name(model, name)
            return mapped.startswith(prefix)

        expert_filter = _ExpertRangeFilter(
            live_experts.num_experts,
            ep_size,
            ep_rank,
            self.model_config,
            self.load_config.parallelism_config,
            placement_overrides={layer_id: placement},
        )
        checkpoint_files = self._checkpoint_files()
        selected_files = weight_mapper.select_safetensor_files(
            self._resolved_model_path(),
            checkpoint_files,
            target_layer_filter,
            self._safetensor_catalog(),
        )
        if not selected_files:
            raise FileNotFoundError(
                f"No checkpoint shards contain MoE weights for layer {layer_id}"
            )
        name_filter = self._checkpoint_name_filter(target_layer_filter, expert_filter)
        read_spec_resolver = self._checkpoint_read_spec_resolver(
            model, target_layer_filter, expert_filter
        )
        method = self._selected_load_method
        if method is None:
            method = self._select_load_method(
                self._resolve_load_method(), model, selected_files, expert_filter
            )

        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            # Online EPLB cannot reuse the engine's WORLD communicator without
            # racing inference collectives.  A rank-local FST group avoids that
            # race, but still stages an entire checkpoint shard on each GPU
            # (8 GiB for the DeepSeek-V2-Lite checkpoint).  Dynamic EPLB only
            # needs one layer, so execute its ReadSpecs through direct bounded
            # safetensors slices instead.  Initial model loading remains FST.
            logger.info(
                "NewLoader dynamic EPLB reload is using memory-bounded "
                "ReadSpec scratch slices instead of online FST staging"
            )
            method = NewLoaderLoadMethod.SCRATCH

        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            self._prepare_model_for_fastsafetensors(reload_experts, reload_device)
            weights = self._fastsafetensors_weights_iterator(
                selected_files,
                reload_device,
                name_filter,
                expert_filter,
                read_spec_resolver,
            )
        else:
            weights = weight_mapper.get_all_weights(
                selected_files,
                device="cpu",
                name_filter=name_filter,
                safetensor_slice_expander=expert_filter,
                read_spec_resolver=read_spec_resolver,
            )

        try:
            for name, tensor in weights:
                mapped_name = self._mapped_checkpoint_name(model, name)
                if not mapped_name.startswith(prefix):
                    raise RuntimeError(
                        f"Dynamic EPLB received non-layer tensor {name!r} "
                        f"mapped to {mapped_name!r}"
                    )
                reload_experts.load_weights({mapped_name[len(prefix) :]: tensor})
        finally:
            close_weights = getattr(weights, "close", None)
            if callable(close_weights):
                close_weights()

        reload_experts.validate_weights_loaded()
        if method == NewLoaderLoadMethod.SCRATCH:
            reload_experts.to(reload_device)
        result = reload_experts.prepare_eplb_reload_weights()
        logger.info(
            "NewLoader reloaded EPLB layer %d on EP rank %d with %s",
            layer_id,
            ep_rank,
            method.value,
        )
        # Return detached CPU tensors so background-thread CUDA postprocessing
        # is complete before the all-rank ready flag is published.  The C++
        # request boundary then installs this exact runtime layout into the
        # Python model's stable device tensor addresses.
        return {
            name: tensor.detach().to(device="cpu").contiguous()
            for name, tensor in result.items()
        }

    def _resolve_load_method(self) -> NewLoaderLoadMethod:
        configured = self.load_config.load_method
        if configured == NewLoaderLoadMethod.AUTO:
            env_method = os.environ.get("LOAD_METHOD", "").strip().lower()
            if env_method:
                try:
                    configured = NewLoaderLoadMethod(env_method)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported LOAD_METHOD environment value {env_method!r}"
                    ) from exc
        return configured

    def _model_type(self) -> str:
        value = (
            self.model_config.get("model_type", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_type", "")
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_config.model_type must be a non-empty string")
        return value.strip()

    def _resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        value = (
            self.model_config.get("model_path", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_path", "")
        )
        if not value:
            raise ValueError("model_path is required")
        return value

    def _checkpoint_files(self):
        if self._ckpt_files is None:
            catalog = self._safetensor_catalog()
            if catalog is not None:
                self._ckpt_files = list(catalog.shard_files)
            else:
                self._ckpt_files = weight_mapper.discover_ckpt_files(
                    self._resolved_model_path(),
                    tp_rank=self.load_config.tp_rank,
                    tp_size=self.load_config.tp_size,
                )
            if not self._ckpt_files:
                raise FileNotFoundError(
                    f"No supported checkpoint files in {self._resolved_model_path()}"
                )
        return self._ckpt_files

    def _safetensor_catalog(self) -> Optional[weight_mapper.CheckpointCatalog]:
        if self._safetensor_catalog_cache is _CHECKPOINT_CATALOG_UNSET:
            self._safetensor_catalog_cache = weight_mapper.safetensor_catalog(
                self._resolved_model_path()
            )
        return self._safetensor_catalog_cache

    def _expert_filter(self):
        if self.load_config.ep_size == 1:
            return None
        num_experts = (
            self.model_config.get("num_experts", 0)
            if isinstance(self.model_config, dict)
            else getattr(
                self.model_config,
                "expert_num",
                getattr(self.model_config, "num_experts", 0),
            )
        )
        if isinstance(num_experts, bool) or not isinstance(num_experts, int):
            raise TypeError(
                f"model_config num_experts must be an integer, got {num_experts!r}"
            )
        if num_experts < 0:
            raise ValueError(
                f"model_config num_experts must be non-negative, got {num_experts}"
            )
        if num_experts == 0:
            model_type = (
                self.model_config.get("model_type", "")
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "model_type", "")
            )
            moe_top_k = (
                self.model_config.get(
                    "moe_k", self.model_config.get("num_experts_per_tok", 0)
                )
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_k", 0)
            )
            moe_style = (
                self.model_config.get("moe_style", 0)
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_style", 0)
            )
            for name, value in (("moe_k", moe_top_k), ("moe_style", moe_style)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"model_config {name} must be an integer, got {value!r}"
                    )
                if value < 0:
                    raise ValueError(
                        f"model_config {name} must be non-negative, got {value}"
                    )
            if "moe" in str(model_type).lower() or moe_top_k > 0 or moe_style > 0:
                raise ValueError(
                    "EP is configured for a MoE model "
                    f"(model_type={model_type!r}, moe_k={moe_top_k}, "
                    f"moe_style={moe_style}), but "
                    "model_config reports zero experts"
                )
            logger.warning(
                "EP size %d is configured, but model_config reports zero experts; "
                "model_type=%r and zero moe_k/moe_style markers identify a dense "
                "model, so expert checkpoint filtering is disabled",
                self.load_config.ep_size,
                model_type,
            )
            return None
        return _ExpertRangeFilter(
            num_experts,
            self.load_config.ep_size,
            self.load_config.ep_rank,
            self.model_config,
            self.load_config.parallelism_config,
        )

    @staticmethod
    def _model_checkpoint_name_filter(
        model: RtpModule,
    ) -> Optional[Callable[[str], bool]]:
        return _require_callable_name_filter(
            model.checkpoint_weight_name_filter(),
            f"{type(model).__name__}.checkpoint_weight_name_filter() return value",
        )

    @staticmethod
    def _checkpoint_name_filter(
        model_filter: Optional[Callable[[str], bool]],
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> Optional[Callable[[str], bool]]:
        model_filter = _require_callable_name_filter(model_filter, "model_filter")
        if model_filter is None and expert_filter is None:
            return None

        def should_load(name: str) -> bool:
            if model_filter is not None and not model_filter(name):
                return False
            if expert_filter is None:
                return True
            if expert_filter.handles_safetensor(name):
                return True
            return expert_filter.should_load(name)

        return should_load

    @staticmethod
    def _mapped_checkpoint_name(model: RtpModule, name: str) -> str:
        mapper = getattr(model, "WEIGHTS_MAPPER", None)
        map_name = getattr(mapper, "map_name", None)
        return map_name(name) if callable(map_name) else name

    def _checkpoint_read_spec_resolver(
        self,
        model: RtpModule,
        model_filter: Optional[Callable[[str], bool]],
        expert_filter: Optional[_ExpertRangeFilter],
        checkpoint_load_plan: Optional[Dict[str, CheckpointLoadTarget]] = None,
    ) -> Callable[[str, Tuple[int, ...]], ReadSpec]:
        model_filter = _require_callable_name_filter(model_filter, "model_filter")
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = max(self.load_config.tp_size, self.load_config.ep_size)

        def resolve(name: str, shape: Tuple[int, ...]) -> ReadSpec:
            shape = tuple(shape)
            if model_filter is not None and not model_filter(name):
                return ReadSpec.skip(name, shape)

            mapped_name = self._mapped_checkpoint_name(model, name)
            load_target, read_spec = model.resolve_checkpoint_load_plan(
                mapped_name, shape
            )
            if checkpoint_load_plan is not None and load_target is not None:
                checkpoint_load_plan[mapped_name] = load_target
            if not model.allow_checkpoint_read_spec(name):
                read_spec = None
            if read_spec is None:
                read_spec = ReadSpec.full(name, shape)
            else:
                read_spec = read_spec.for_checkpoint_name(name)

            if expert_filter is not None:
                target_ranks = expert_filter.target_world_ranks(name, world_size)
                if target_ranks is not None:
                    read_spec = read_spec.with_targets(
                        target_ranks, expert_filter.should_load(name)
                    )
            return read_spec

        return resolve

    def _validate_ep_checkpoint_format(
        self,
        checkpoint_files,
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> None:
        if expert_filter is None:
            return
        unsupported = [
            path
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if unsupported:
            raise ValueError(
                "EP streaming requires safetensors checkpoints with per-expert "
                "keys; PyTorch checkpoints are deserialized as a whole before "
                f"name filtering and would materialize non-local experts: {unsupported}"
            )

    def _create_model(self) -> nn.Module:
        model_type = self._model_type()
        try:
            model_cls = get_model_class(model_type)
        except KeyError as exc:
            raise ValueError(
                f"Model type {model_type!r} is not registered; available={list_models()}"
            ) from exc
        model = model_cls(self.model_config, self.load_config)
        if not isinstance(model, RtpModule):
            raise TypeError(
                f"Registered model {model_cls.__name__} must inherit RtpModule to "
                "provide newloader completeness validation"
            )
        if self.load_config.custom_weight_mappings and not bool(
            getattr(model, "supports_custom_weight_mappings", False)
        ):
            raise NotImplementedError(
                f"{type(model).__name__} does not support downstream custom weights"
            )
        try:
            source = inspect.getfile(model_cls)
        except (TypeError, OSError):
            source = "<unknown>"
        logger.info(
            "Created newloader model %s from %s", model_cls.__qualname__, source
        )
        return model

    @staticmethod
    def _validate_loaded_weights(model: nn.Module) -> None:
        loaded_tensor_ids = frozenset(collect_loaded_tensor_ids(model))
        root_validator = getattr(model, "validate_weights_loaded", None)
        if not callable(root_validator):
            raise TypeError(
                f"Registered model {type(model).__name__} has no weight completeness validator"
            )
        root_validator(loaded_tensor_ids)

        for module_name, module in model.named_modules():
            if module is model:
                continue
            custom_loader = getattr(type(module), "load_weights", None)
            if (
                custom_loader is not None
                and custom_loader is not RtpModule.load_weights
            ):
                validator = getattr(module, "validate_weights_loaded", None)
                if not callable(validator):
                    raise TypeError(
                        f"Custom weight loader {type(module).__name__}.load_weights() "
                        "must define validate_weights_loaded()"
                    )
                try:
                    validator(loaded_tensor_ids)
                except Exception as exc:
                    context = (
                        f"Weight validation failed for {module_name} "
                        f"({type(module).__name__})"
                    )
                    if exc.args:
                        exc.args = (f"{context}: {exc.args[0]}", *exc.args[1:])
                    else:
                        exc.args = (context,)
                    raise

    @staticmethod
    def _run_post_load_hooks(model: nn.Module) -> None:
        for module in model.modules():
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    @staticmethod
    def _prepare_model_for_fastsafetensors(
        model: nn.Module,
        target_device: str,
        deferred_modules: Optional[List[RtpModule]] = None,
    ) -> None:
        """Allocate checkpoint-backed parameters without copying placeholders.

        NewLoader constructs the model on CPU so capability checks and AUTO
        selection do not consume accelerator memory.  A normal ``model.to``
        would then copy every uninitialized parameter to the accelerator just
        before fastsafetensors overwrites it.  Allocate fresh parameter storage
        on the target instead, while retaining the values of buffers that may
        be derived by the constructor rather than supplied by the checkpoint.

        ``RtpModule._apply`` memoizes shared tensors and restores aliases, so
        tied parameters and parameter/buffer aliases remain intact.
        """
        device = torch.device(target_device)
        deferred_tensor_ids = set()
        for module in deferred_modules or ():
            deferred_tensor_ids.update(
                id(tensor)
                for tensor in list(module.parameters(recurse=True))
                + list(module.buffers(recurse=True))
                if tensor is not None
            )
        parameter_ids = {
            id(parameter)
            for parameter in model.parameters(recurse=True)
            if parameter is not None
        }

        def allocate_parameter_or_move_buffer(tensor: torch.Tensor) -> torch.Tensor:
            if id(tensor) in deferred_tensor_ids:
                return tensor
            if id(tensor) in parameter_ids:
                return torch.empty_like(tensor, device=device)
            return tensor.to(device=device)

        model._apply(allocate_parameter_or_move_buffer)

    @staticmethod
    def _requires_staged_device_postprocess(model: nn.Module) -> bool:
        return any(
            isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
            for module in model.modules()
        )

    @staticmethod
    def _staged_device_modules(model: nn.Module) -> List[RtpModule]:
        return [
            module
            for module in model.modules()
            if isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
        ]

    @staticmethod
    def _validate_staged_device_modules(
        model: nn.Module,
        staged_modules: List[RtpModule],
        *,
        require_load_complete: bool = False,
    ) -> None:
        if not staged_modules:
            return
        staged_set = set(staged_modules)
        tensor_registrations = {}
        for module in model.modules():
            tensors = list(
                module.named_parameters(recurse=False, remove_duplicate=False)
            ) + list(module.named_buffers(recurse=False, remove_duplicate=False))
            for name, tensor in tensors:
                if tensor is not None:
                    tensor_registrations.setdefault(id(tensor), []).append(
                        (module, name)
                    )
        for registrations in tensor_registrations.values():
            if (
                any(module in staged_set for module, _ in registrations)
                and len(registrations) > 1
            ):
                aliases = sorted(
                    f"{type(module).__name__}.{name}" for module, name in registrations
                )
                raise RuntimeError(
                    "Online-quantized staging does not support shared tensor "
                    f"aliases: {aliases}"
                )
        for module in staged_modules:
            if any(True for _ in module.children()):
                raise RuntimeError(
                    "Staged device postprocess is supported only for leaf modules, "
                    f"got {type(module).__name__}"
                )
            if require_load_complete and (
                type(module).checkpoint_load_complete
                is RtpModule.checkpoint_load_complete
            ):
                raise RuntimeError(
                    f"Staged leaf {type(module).__name__} does not implement "
                    "checkpoint_load_complete()"
                )

    @staticmethod
    def _fastsafetensors_api():
        try:
            from fastsafetensors import ParallelLoader, SingleGroup
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                "the fastsafetensors package is unavailable or failed to import: "
                f"{exc}"
            ) from exc

        for symbol_name, symbol in (
            ("ParallelLoader", ParallelLoader),
            ("SingleGroup", SingleGroup),
        ):
            if not callable(symbol):
                raise RuntimeError(f"fastsafetensors.{symbol_name} must be callable")
        try:
            parameters = inspect.signature(ParallelLoader.__init__).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot inspect fastsafetensors.ParallelLoader constructor"
            ) from exc
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        missing = _FASTSAFETENSORS_REQUIRED_ARGUMENTS - set(parameters)
        if missing and not accepts_kwargs:
            raise RuntimeError(
                "fastsafetensors.ParallelLoader is incompatible with NewLoader; "
                f"missing constructor arguments {sorted(missing)}"
            )
        return ParallelLoader, SingleGroup

    @staticmethod
    def _safetensor_metadata_summary(
        model_path: str,
        checkpoint_files: List[str],
        catalog: Optional[weight_mapper.CheckpointCatalog] = None,
    ) -> Tuple[List[str], List[str]]:
        """Return duplicate names and stacked expert names without reading data.

        A valid Hugging Face index is the preferred metadata source.  The
        streaming iterator still detects duplicate names while loading, so an
        index avoids a second full header walk without weakening correctness.
        Header inspection remains the fallback for unindexed checkpoints.
        """
        selected_identities = {
            os.path.realpath(os.fspath(path)) for path in checkpoint_files
        }
        index_entries = weight_mapper.safetensor_index_entries(model_path, catalog)
        if index_entries is not None:
            identity_by_shard = {
                shard_path: os.path.realpath(shard_path)
                for shard_path in dict.fromkeys(path for _, path in index_entries)
            }
            indexed_identities = set(identity_by_shard.values())
            if selected_identities.issubset(indexed_identities):
                stacked = [
                    name
                    for name, shard_path in index_entries
                    if identity_by_shard[shard_path] in selected_identities
                    and _STACKED_EXPERT_RE.fullmatch(name)
                ]
                return [], stacked

        from safetensors import safe_open

        seen = set()
        duplicates = set()
        stacked = []
        for path in checkpoint_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        duplicates.add(name)
                    seen.add(name)
                    if _STACKED_EXPERT_RE.fullmatch(name):
                        stacked.append(name)
        return sorted(duplicates), sorted(stacked)

    def _fastsafetensors_distributed_reason(
        self, expert_filter: Optional[_ExpertRangeFilter] = None
    ) -> Optional[str]:
        """Validate WORLD and rank fields before choosing an FST subgroup."""
        parallelism = self.load_config.parallelism_config
        expected_world_size = getattr(
            parallelism,
            "world_size",
            max(self.load_config.tp_size, self.load_config.ep_size),
        )
        if (
            isinstance(expected_world_size, bool)
            or not isinstance(expected_world_size, int)
            or expected_world_size <= 0
        ):
            return f"invalid configured world_size={expected_world_size!r}"
        if (
            expert_filter is not None
            and self.load_config.ep_size != expected_world_size
        ):
            return (
                "EP fastsafetensors loading requires ep_size to span WORLD: "
                f"ep_size={self.load_config.ep_size}, "
                f"world_size={expected_world_size}"
            )
        if not torch.distributed.is_initialized():
            if expected_world_size > 1:
                return (
                    "TP/EP fastsafetensors loading requires an initialized "
                    f"distributed WORLD group of size {expected_world_size}"
                )
            return None

        world_size = torch.distributed.get_world_size()
        if world_size != expected_world_size:
            return (
                "distributed WORLD group does not match the configured "
                f"topology: world_size={world_size}, expected={expected_world_size}"
            )
        world_rank = torch.distributed.get_rank()
        expected_tp_rank = world_rank % self.load_config.tp_size
        expected_ep_rank = world_rank % self.load_config.ep_size
        if (
            self.load_config.tp_rank != expected_tp_rank
            or self.load_config.ep_rank != expected_ep_rank
        ):
            return (
                "distributed WORLD rank does not match the NewLoader TP/EP "
                f"ranks: world_rank={world_rank}, loader_tp_rank="
                f"{self.load_config.tp_rank}, expected_tp_rank={expected_tp_rank}, "
                f"loader_ep_rank={self.load_config.ep_rank}, "
                f"expected_ep_rank={expected_ep_rank}"
            )
        return None

    def _fastsafetensors_process_group(
        self,
        SingleGroup,
        checkpoint_files: List[str],
        expert_filter: Optional[_ExpertRangeFilter],
    ):
        """Return the smallest group containing every checkpoint consumer."""
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files):
            logger.info("NewLoader fastsafetensors process group: rank-local")
            return SingleGroup()
        if not torch.distributed.is_initialized():
            logger.info("NewLoader fastsafetensors process group: single-rank")
            return SingleGroup()
        if expert_filter is not None:
            logger.info("NewLoader fastsafetensors process group: EP WORLD")
            return torch.distributed.group.WORLD
        if self.load_config.tp_size <= 1:
            logger.info("NewLoader fastsafetensors process group: DP-local rank")
            return SingleGroup()
        world_size = torch.distributed.get_world_size()
        if world_size == self.load_config.tp_size:
            logger.info("NewLoader fastsafetensors process group: TP WORLD")
            return torch.distributed.group.WORLD
        from rtp_llm.models_py.distributed.collective_torch import Group, _get_group

        process_group = _get_group(Group.TP)
        group_size = torch.distributed.get_world_size(process_group)
        group_rank = torch.distributed.get_rank(process_group)
        if group_size != self.load_config.tp_size:
            raise RuntimeError(
                "FST TP process group size does not match NewLoader: "
                f"group_size={group_size}, tp_size={self.load_config.tp_size}"
            )
        if group_rank != self.load_config.tp_rank:
            raise RuntimeError(
                "FST TP process group rank does not match NewLoader: "
                f"group_rank={group_rank}, tp_rank={self.load_config.tp_rank}"
            )
        logger.info(
            "NewLoader fastsafetensors process group: TP subgroup rank=%d size=%d",
            group_rank,
            group_size,
        )
        return process_group

    @staticmethod
    def _fastsafetensors_bbuf_size_kb(checkpoint_files: List[str]) -> int:
        max_shard_bytes = max(os.path.getsize(path) for path in checkpoint_files)
        max_shard_kb = (max_shard_bytes + 1023) // 1024
        return min(
            _FASTSAFETENSORS_MAX_BBUF_SIZE_KB,
            max(_FASTSAFETENSORS_MIN_BBUF_SIZE_KB, max_shard_kb),
        )

    @staticmethod
    def _cuda_cufile_available() -> bool:
        """Use fastsafetensors' native probe as the GDS source of truth."""
        try:
            from fastsafetensors import cpp as fastsafetensors_cpp
        except (ImportError, OSError, RuntimeError):
            return False
        try:
            return bool(fastsafetensors_cpp.is_cufile_found())
        except (AttributeError, OSError, RuntimeError):
            return False

    @staticmethod
    def _accelerator_backend_name() -> str:
        """Identify CUDA-compatible runtimes without importing the device package."""
        if torch.version.hip is not None:
            return "ROCm"
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return "Ppu"
        return "Cuda"

    @staticmethod
    def _fastsafetensors_use_nogds(device: torch.device) -> bool:
        if device.type == "cpu":
            return True
        if os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1":
            return True
        if (
            device.type == "cuda"
            and torch.version.hip is None
            and not NewModelLoader._cuda_cufile_available()
        ):
            logger.warning(
                "libcufile.so is unavailable; NewLoader fastsafetensors is "
                "falling back to the CUDA nogds path"
            )
            return True
        return False

    @staticmethod
    def _fastsafetensors_backend_options(
        device: torch.device,
    ) -> Tuple[bool, bool]:
        """Return ``(nogds, use_shm)`` for the active accelerator backend.

        ``SafeTensorsFileLoader`` checks ``use_shm`` before ``nogds`` and GDS.
        Therefore CUDA GDS must set both flags to false; enabling ``use_shm``
        would silently select the shared-memory reader instead of GDS.
        """
        use_nogds = NewModelLoader._fastsafetensors_use_nogds(device)
        use_shm = (
            device.type == "cuda" and torch.version.hip is not None and not use_nogds
        )
        return use_nogds, use_shm

    @staticmethod
    def _model_storage_bytes(model: nn.Module) -> int:
        seen = set()
        total = 0
        tensors = list(
            model.named_parameters(recurse=True, remove_duplicate=False)
        ) + list(model.named_buffers(recurse=True, remove_duplicate=False))
        for _, tensor in tensors:
            if tensor is None or id(tensor) in seen:
                continue
            if tensor.is_meta:
                raise RuntimeError(
                    "model contains meta tensors that cannot be sized for automatic "
                    "fastsafetensors selection"
                )
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
        return total

    def _fastsafetensors_prerequisite_reason(
        self,
        model: nn.Module,
        checkpoint_files: List[str],
        *,
        automatic: bool,
        expert_filter: Optional[_ExpertRangeFilter] = None,
    ) -> Optional[str]:
        if not checkpoint_files:
            return "checkpoint shard selection is empty"
        non_safetensors = [
            os.fspath(path)
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if non_safetensors:
            return f"checkpoint contains non-safetensors files {non_safetensors}"
        rank_local_checkpoint = weight_mapper.is_rank_local_checkpoint(checkpoint_files)
        if rank_local_checkpoint and not bool(
            getattr(model, "supports_rank_local_checkpoint", False)
        ):
            return (
                f"{type(model).__name__} does not support rank-local consolidated "
                "checkpoints"
            )
        staged_modules = self._staged_device_modules(model)
        try:
            self._validate_staged_device_modules(
                model, staged_modules, require_load_complete=True
            )
        except (RuntimeError, TypeError, ValueError) as exc:
            return f"staged device postprocessing is incompatible with FST: {exc}"

        target_device = torch.device(self.load_config.device)
        if target_device.type not in ("cpu", "cuda"):
            return f"runtime device {target_device} is not supported"
        if automatic and target_device.type == "cpu":
            return (
                "the runtime device is CPU, where direct accelerator loading "
                "has no benefit"
            )
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                return f"runtime device {target_device} is unavailable"
            backend = self._accelerator_backend_name()
            if automatic and backend != "Cuda":
                return (
                    "automatic fastsafetensors selection is disabled on the "
                    f"unvalidated {backend} backend; request "
                    "fastsafetensors explicitly to override"
                )
            machine = platform.machine().lower()
            if automatic and machine not in ("amd64", "x86_64"):
                return (
                    "automatic fastsafetensors selection is disabled on the "
                    f"unvalidated {machine or 'unknown'} architecture; request "
                    "fastsafetensors explicitly to override"
                )
            if (
                torch.version.hip is not None
                and os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1"
            ):
                return (
                    "FASTSAFETENSORS_NOGDS=1 is unsupported on ROCm by the "
                    "installed fastsafetensors backend"
                )
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            device_count = torch.cuda.device_count()
            if not 0 <= device_index < device_count:
                return (
                    f"runtime device index {device_index} is outside the available "
                    f"CUDA device range [0, {device_count})"
                )

        distributed_reason = self._fastsafetensors_distributed_reason(expert_filter)
        if distributed_reason is not None:
            return distributed_reason

        try:
            ParallelLoader, _ = self._fastsafetensors_api()
        except RuntimeError as exc:
            return str(exc)
        try:
            duplicates, stacked_experts = self._safetensor_metadata_summary(
                self._resolved_model_path(),
                checkpoint_files,
                self._safetensor_catalog(),
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return f"failed to inspect safetensors metadata: {exc}"
        if duplicates:
            sample = duplicates[:10]
            suffix = (
                f" (+{len(duplicates) - len(sample)} more)"
                if len(duplicates) > 10
                else ""
            )
            return f"checkpoint tensor names are duplicated: {sample}{suffix}"
        if expert_filter is not None and stacked_experts:
            expected_world_size = max(
                self.load_config.tp_size, self.load_config.ep_size
            )
            if self.load_config.ep_size != expected_world_size:
                return (
                    "stacked all-expert FST loading requires ep_size to equal "
                    "the WORLD size so get_sharded(dim=0) maps one expert range "
                    "to each EP rank"
                )
            if not hasattr(ParallelLoader, "_consume_single_batch"):
                return (
                    "the installed fastsafetensors ParallelLoader does not expose "
                    "the batch hook required for stacked all-expert sharding"
                )
        if automatic:
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device_index)
                model_bytes = self._model_storage_bytes(model)
                max_shard_bytes = max(
                    os.path.getsize(path) for path in checkpoint_files
                )
                bbuf_bytes = self._fastsafetensors_bbuf_size_kb(checkpoint_files) * 1024
            except (OSError, RuntimeError, ValueError) as exc:
                return f"cannot determine fastsafetensors memory safety: {exc}"
            required_bytes = model_bytes + 3 * max(max_shard_bytes, bbuf_bytes)
            logger.info(
                "NewLoader fastsafetensors memory check: free=%.2fGiB, "
                "model=%.2fGiB, max_shard=%.2fGiB, bbuf=%.2fGiB, "
                "required=%.2fGiB",
                free_bytes / (1024**3),
                model_bytes / (1024**3),
                max_shard_bytes / (1024**3),
                bbuf_bytes / (1024**3),
                required_bytes / (1024**3),
            )
            if free_bytes <= required_bytes:
                return (
                    f"insufficient accelerator memory: free={free_bytes} bytes, "
                    f"required>{required_bytes} bytes"
                )
        return None

    @staticmethod
    def _synchronize_fastsafetensors_prerequisite(
        local_reason: Optional[str],
    ) -> Optional[str]:
        """Require every distributed rank to make the same loader decision."""
        if not torch.distributed.is_initialized():
            return local_reason
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return local_reason
        reasons: List[Optional[str]] = [None] * world_size
        try:
            torch.distributed.all_gather_object(reasons, local_reason)
        except (RuntimeError, TypeError, ValueError) as exc:
            return (
                "cannot synchronize fastsafetensors prerequisites across ranks: "
                f"{exc}"
            )
        failures = [
            f"rank {rank}: {reason}"
            for rank, reason in enumerate(reasons)
            if reason is not None
        ]
        if not failures:
            return None
        sample = failures[:4]
        suffix = (
            f" (+{len(failures) - len(sample)} more)"
            if len(failures) > len(sample)
            else ""
        )
        return f"distributed prerequisite failure: {'; '.join(sample)}{suffix}"

    def _select_load_method(
        self,
        requested: NewLoaderLoadMethod,
        model: nn.Module,
        checkpoint_files: List[str],
        expert_filter: Optional[_ExpertRangeFilter] = None,
    ) -> NewLoaderLoadMethod:
        if requested == NewLoaderLoadMethod.SCRATCH:
            return requested
        automatic = requested == NewLoaderLoadMethod.AUTO
        reason = self._fastsafetensors_prerequisite_reason(
            model,
            checkpoint_files,
            automatic=automatic,
            expert_filter=expert_filter,
        )
        reason = self._synchronize_fastsafetensors_prerequisite(reason)
        if reason is None:
            selected = NewLoaderLoadMethod.FASTSAFETENSORS
            logger.info(
                "NewLoader selected fastsafetensors (%s)",
                "automatic" if automatic else "explicit",
            )
            return selected
        if automatic:
            logger.info(
                "NewLoader automatic fastsafetensors selection fell back to "
                "scratch: %s",
                reason,
            )
            return NewLoaderLoadMethod.SCRATCH
        raise RuntimeError(f"NewLoader fastsafetensors prerequisite failed: {reason}")

    def _fastsafetensors_weights_iterator(
        self,
        checkpoint_files: List[str],
        device: str,
        name_filter: Optional[Callable[[str], bool]],
        safetensor_slice_expander: Optional[_ExpertRangeFilter],
        read_spec_resolver: Optional[Callable[[str, Tuple[int, ...]], ReadSpec]] = None,
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream safetensors directly to the model's runtime device.

        The yielded tensor is consumed synchronously by ``RtpModule.load_weights``
        before the iterator advances. Compatible FST versions execute ReadSpecs
        in their consumer so TP shards and EP expert ranges are materialized
        before recursive model dispatch.
        """
        ParallelLoader, SingleGroup = self._fastsafetensors_api()
        process_group = self._fastsafetensors_process_group(
            SingleGroup, checkpoint_files, safetensor_slice_expander
        )

        target_device = torch.device(device)
        loader_device = str(target_device)
        if target_device.type == "cuda" and target_device.index is None:
            loader_device = f"cuda:{torch.cuda.current_device()}"
        use_nogds, use_shm = NewModelLoader._fastsafetensors_backend_options(
            target_device
        )
        bbuf_size_kb = NewModelLoader._fastsafetensors_bbuf_size_kb(checkpoint_files)
        loader_kwargs = dict(
            pg=process_group,
            hf_weights_files=sorted(os.fspath(path) for path in checkpoint_files),
            use_tqdm_on_load=True,
            device=loader_device,
            bbuf_size_kb=bbuf_size_kb,
            use_shm=use_shm,
            nogds=use_nogds,
        )
        loader_handles_read_specs = False
        if read_spec_resolver is not None and hasattr(
            ParallelLoader, "_consume_single_batch"
        ):
            try:
                from fastsafetensors.parallel_loader import TimingContext
                from rtp_llm.models_py.fastsafetensors_read_spec import (
                    build_read_spec_parallel_loader,
                )

                read_spec_loader = build_read_spec_parallel_loader(
                    ParallelLoader, TimingContext
                )
                loader = read_spec_loader(
                    read_spec_resolver=read_spec_resolver,
                    safetensor_slice_expander=safetensor_slice_expander,
                    **loader_kwargs,
                )
                loader_handles_read_specs = True
            except (ImportError, TypeError, ValueError) as exc:
                if safetensor_slice_expander is not None:
                    raise RuntimeError(
                        "fastsafetensors does not expose compatible internals "
                        "required for EP stacked expert sharding"
                    ) from exc
                logger.warning(
                    "fastsafetensors does not expose compatible ReadSpec "
                    "internals; applying slices after tensor materialization"
                )
                loader = ParallelLoader(**loader_kwargs)
        else:
            loader = ParallelLoader(**loader_kwargs)
        seen = set()
        try:
            for name, checkpoint_value in loader.iterate_weights():
                if isinstance(checkpoint_value, LoadedCheckpointTensor):
                    tensor_value = checkpoint_value.tensor
                else:
                    tensor_value = checkpoint_value
                if read_spec_resolver is not None and not loader_handles_read_specs:
                    read_spec = read_spec_resolver(name, tuple(tensor_value.shape))
                    if not read_spec.load:
                        continue
                    checkpoint_value = read_spec.materialize(tensor_value)
                if name_filter is not None and not name_filter(name):
                    continue
                if (
                    safetensor_slice_expander is not None
                    and safetensor_slice_expander.handles_safetensor(name)
                ):
                    raise RuntimeError(
                        "EP fastsafetensors backend failed to shard and expand "
                        f"stacked all-expert tensor {name!r}"
                    )
                if name in seen:
                    raise RuntimeError(
                        f"Checkpoint tensor {name!r} appears more than once during "
                        "fastsafetensors loading"
                    )
                seen.add(name)
                yield name, checkpoint_value
        finally:
            backing_loader = getattr(loader, "loader", None)
            close = getattr(backing_loader, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_runtime_backends(model: nn.Module, device: str) -> None:
        target_device = torch.device(device)
        for module in model.modules():
            if isinstance(module, RtpModule):
                module.validate_runtime_device(target_device)

    @staticmethod
    def _migrate_staged_modules(model: nn.Module, device: str) -> None:
        """Move and compress online-quantized leaves one at a time.

        Migrating the entire model first would retain every BF16 staging weight on
        the accelerator until post-load hooks run. Processing each marked leaf
        immediately bounds the extra accelerator memory to one staging layer.
        """
        target_device = torch.device(device)
        staged_modules = NewModelLoader._staged_device_modules(model)
        NewModelLoader._validate_staged_device_modules(model, staged_modules)
        for module in staged_modules:
            module.to(target_device)
            module.process_weights_after_loading()

    @torch.inference_mode()
    def load(self) -> nn.Module:
        requested_method = self._resolve_load_method()
        checkpoint_files = self._checkpoint_files()
        expert_filter = self._expert_filter()
        self._validate_ep_checkpoint_format(checkpoint_files, expert_filter)
        with torch.device("cpu"):
            model = self._create_model()
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files) and not bool(
            getattr(model, "supports_rank_local_checkpoint", False)
        ):
            raise ValueError(
                f"{type(model).__name__} does not support rank-local consolidated "
                "checkpoints; use a global HF checkpoint"
            )
        self._validate_runtime_backends(model, self.load_config.device)
        model_filter = self._model_checkpoint_name_filter(model)
        name_filter = self._checkpoint_name_filter(model_filter, expert_filter)
        checkpoint_load_plan: Dict[str, CheckpointLoadTarget] = {}
        read_spec_resolver = self._checkpoint_read_spec_resolver(
            model, model_filter, expert_filter, checkpoint_load_plan
        )
        selected_files = weight_mapper.select_safetensor_files(
            self._resolved_model_path(),
            checkpoint_files,
            model_filter,
            self._safetensor_catalog(),
        )
        if len(selected_files) != len(checkpoint_files):
            logger.info(
                "Model checkpoint filter selected %d/%d shards",
                len(selected_files),
                len(checkpoint_files),
            )
        method = self._select_load_method(
            requested_method,
            model,
            selected_files,
            expert_filter,
        )
        self._selected_load_method = method
        logger.info(
            "NewLoader load method: requested=%s, selected=%s",
            requested_method.value,
            method.value,
        )
        started = time.time()
        staged_coordinator = None
        coordinator_modules: List[RtpModule] = []
        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            staged_modules = self._staged_device_modules(model)
            self._validate_staged_device_modules(
                model, staged_modules, require_load_complete=True
            )
            self._prepare_model_for_fastsafetensors(
                model,
                self.load_config.device,
                deferred_modules=staged_modules,
            )
            if staged_modules:
                staged_coordinator = _StagedFSTLoadCoordinator(
                    model, staged_modules, self.load_config.device
                )
                coordinator_modules = [
                    module
                    for module in model.modules()
                    if isinstance(module, RtpModule)
                ]
                for module in coordinator_modules:
                    module._rtp_staged_load_coordinator = staged_coordinator
        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            weights = self._fastsafetensors_weights_iterator(
                selected_files,
                self.load_config.device,
                name_filter,
                expert_filter,
                read_spec_resolver,
            )
        else:
            weights = weight_mapper.get_all_weights(
                selected_files,
                device="cpu",
                name_filter=name_filter,
                safetensor_slice_expander=expert_filter,
                read_spec_resolver=read_spec_resolver,
            )
        previous_load_plan = getattr(
            model, "_rtp_checkpoint_load_plan", _CHECKPOINT_LOAD_PLAN_UNSET
        )
        model._rtp_checkpoint_load_plan = checkpoint_load_plan
        try:
            model.load_weights(weights)
            if staged_coordinator is not None:
                staged_coordinator.finish()
        finally:
            if previous_load_plan is _CHECKPOINT_LOAD_PLAN_UNSET:
                delattr(model, "_rtp_checkpoint_load_plan")
            else:
                model._rtp_checkpoint_load_plan = previous_load_plan
            if staged_coordinator is not None:
                for module in coordinator_modules:
                    delattr(module, "_rtp_staged_load_coordinator")
            close_weights = getattr(weights, "close", None)
            if callable(close_weights):
                close_weights()
        self._validate_loaded_weights(model)
        logger.info(
            "Streamed checkpoint tensors with %s in %.2fs",
            method.value,
            time.time() - started,
        )

        if method == NewLoaderLoadMethod.SCRATCH:
            self._migrate_staged_modules(model, self.load_config.device)
        model.to(self.load_config.device)
        self._validate_loaded_weights(model)
        self._run_post_load_hooks(model)
        model.eval()
        return model
