"""Online checkpoint updates for NewLoader's stable parameter/storage owners.

The service must pause every rank before updating and resume only after all
updates succeed. The shared lock also excludes in-flight model forwards.
Incoming names and tensors use global HF checkpoint layout; leaf loaders own
TP slicing and fusion. Each RPC validates/stages before changing live tensors.
"""

import threading
from dataclasses import replace

import torch

from rtp_llm.models_py.runtime_weight_update import (
    RuntimeWeightTransaction,
    TensorUpdate,
    _signature,
)


class NewLoaderWeightManager:
    requires_paused_engine = True

    def __init__(self, model, loader):
        if not getattr(model, "supports_checkpoint_weight_updates", False):
            raise NotImplementedError(f"{type(model).__name__} has no checkpoint update contract")
        if loader.load_config.quant_config.quant_type != "none":
            raise NotImplementedError("Quantized NewLoader online updates are not enabled")
        self._model = model
        self._loader = loader
        self._lock = threading.RLock()
        model._weight_update_lock = self._lock
        self.version = 0
        self._parameter_contract = {
            name: (tensor, _signature(tensor))
            for name, tensor in model.named_parameters(remove_duplicate=False)
        }
        self._runtime_contract = {
            name: (tensor, _signature(tensor)) for name, tensor in model.runtime_weight_view().items()
        }

    def _validate_contract(self):
        current = dict(self._model.named_parameters(remove_duplicate=False))
        if current.keys() != self._parameter_contract.keys():
            raise RuntimeError("NewLoader parameter registrations changed after loading")
        for name, (original, signature) in self._parameter_contract.items():
            if current[name] is not original or _signature(current[name]) != signature:
                raise RuntimeError(f"NewLoader parameter/storage changed for {name!r}")
        for name, (tensor, signature) in self._runtime_contract.items():
            if _signature(tensor) != signature:
                raise RuntimeError(f"Exported runtime storage changed for {name!r}")

    @torch.inference_mode()
    def update_tensors(self, weights: dict[str, torch.Tensor]) -> None:
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Online updates require a non-empty tensor mapping")
        with self._lock:
            self._validate_contract()
            devices = {tensor.device for tensor, _ in self._parameter_contract.values() if tensor.is_cuda}
            for device in devices:
                torch.cuda.synchronize(device)
            groups = {}
            for name, tensor in weights.items():
                if not isinstance(name, str) or not name or not isinstance(tensor, torch.Tensor):
                    raise TypeError("Online updates require named checkpoint tensors")
                mapped = self._loader._mapped_checkpoint_name(self._model, name)
                if mapped == "lm_head.weight" and self._model.tie_word_embeddings:
                    raise ValueError("Tied LM head is a non-owning alias; update model.embed_tokens.weight")
                target, _ = self._model.resolve_checkpoint_load_plan(mapped, tuple(tensor.shape))
                if target is None or not callable(getattr(target.module, "prepare_weight_updates", None)):
                    raise NotImplementedError(f"No online checkpoint update contract for {name!r}")
                group = groups.setdefault(target.module, {})
                if target.name in group:
                    raise ValueError(f"Duplicate mapped checkpoint tensor {mapped!r}")
                group[target.name] = tensor.detach().to(device="cpu", copy=True)
                if mapped == "embed_tokens.weight" and self._model.tie_word_embeddings:
                    groups.setdefault(self._model.lm_head, {})["weight"] = group[target.name]

            module_names = {module: name for name, module in self._model.named_modules()}
            updates = []
            for module, incoming in groups.items():
                leaf_updates = module.prepare_weight_updates(incoming)
                if not leaf_updates:
                    raise RuntimeError("Checkpoint conversion produced no runtime updates")
                for update in leaf_updates:
                    source = update.source
                    if module is self._model.lm_head and update.name == "weight":
                        if self._model.normalize_lm_head_weight:
                            source = torch.nn.functional.normalize(source, dim=1)
                        source = source * self._model.logit_scale
                    updates.append(replace(update, name=f"{module_names[module]}.{update.name}", source=source))
            transaction = RuntimeWeightTransaction(updates)
            transaction.install()
            transaction.commit()
            self.version += 1

    def update(self, req: dict[str, str]) -> None:
        for field in ("name", "method", "desc"):
            if not isinstance(req.get(field), str) or not req[field]:
                raise ValueError(f"Missing or invalid online update field {field!r}")
        # Import transport dependencies only for RPC callers. Conversion tests
        # and local update_tensors do not require CUDA IPC runtime libraries.
        if req["method"] == "shm":
            from rtp_llm.model_loader.tipc import SharedMemIpcMeta, SharedMemoryIPCHelper
            tensor = SharedMemoryIPCHelper().build_from_meta(SharedMemIpcMeta.decode(req["desc"]))
        elif req["method"] == "cuda_ipc":
            from rtp_llm.model_loader.tipc import CudaIpcHelper
            tensor = CudaIpcHelper().build_from_meta(bytes.fromhex(req["desc"]))
        else:
            raise ValueError(f"Unsupported online update transport {req['method']!r}")
        self.update_tensors({req["name"]: tensor})
