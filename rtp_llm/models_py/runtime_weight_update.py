"""In-place runtime weight transactions shared by EPLB and online updates.

Callers must exclude inference and other writers until commit or rollback.
Tensor objects, storage addresses, shapes, dtypes and strides never change.
Validation and CPU snapshots finish before the first live write. Snapshots
remain available until a distributed caller acknowledges every rank.
"""

import copy
from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class TensorUpdate:
    name: str
    target: torch.Tensor
    source: torch.Tensor


def prepare_leaf_weight_updates(module, weights, *, reset_fields=()) -> list[TensorUpdate]:
    """Reuse a leaf's checkpoint TP/layout conversion on private CPU storage.

    Only explicitly supported leaves call this helper. Runtime executors and
    post-load transforms must never run on this shallow metadata copy.
    """
    if list(module.children()):
        raise ValueError("Online checkpoint updates require leaf-owned conversion")
    staged = copy.copy(module)
    staged._parameters = {
        name: None if tensor is None else torch.nn.Parameter(
            tensor.detach().to(device="cpu", copy=True), requires_grad=False
        )
        for name, tensor in module._parameters.items()
    }
    staged._buffers = {
        name: None if tensor is None else tensor.detach().to(device="cpu", copy=True)
        for name, tensor in module._buffers.items()
    }
    staged._rtp_loaded_weight_names = set()
    for name in reset_fields:
        value = getattr(module, name, None)
        if isinstance(value, (set, dict)):
            setattr(staged, name, type(value)())
    staged.load_weights(weights)
    changed = staged._rtp_loaded_weight_names | set(getattr(staged, "_loaded_parameter_shards", {}))
    return [
        TensorUpdate(name, module._parameters[name], staged._parameters[name])
        for name in sorted(changed)
        if name in staged._parameters
    ]


def _signature(tensor: torch.Tensor) -> tuple:
    return (
        tensor.device,
        tensor.dtype,
        tuple(tensor.shape),
        tensor.stride(),
        tensor.storage_offset(),
        tensor.untyped_storage().data_ptr(),
        tensor.untyped_storage().nbytes(),
    )


class RuntimeWeightTransaction:
    """Prepare, install and acknowledge one rank's already-materialized weights."""

    @torch.no_grad()
    def __init__(self, updates: Iterable[TensorUpdate]):
        updates = list(updates)
        if not updates:
            raise ValueError("A runtime weight transaction must contain updates")
        names = set()
        storages = set()
        self._devices = set()
        for update in updates:
            if not isinstance(update.name, str) or not update.name:
                raise ValueError("Runtime weight names must be non-empty strings")
            if update.name in names:
                raise ValueError(f"Duplicate runtime weight {update.name!r}")
            names.add(update.name)
            target, source = update.target, update.source
            if not isinstance(target, torch.Tensor) or not isinstance(source, torch.Tensor):
                raise TypeError(f"Runtime weight {update.name!r} requires tensors")
            if target.is_meta or source.is_meta:
                raise ValueError(f"Runtime weight {update.name!r} cannot use meta tensors")
            if target.layout != torch.strided or source.layout != torch.strided:
                raise ValueError(f"Runtime weight {update.name!r} must use strided layout")
            if target.device.type not in ("cpu", "cuda") or source.device.type not in ("cpu", "cuda"):
                raise ValueError("Runtime updates support CPU and CUDA/ROCm tensors")
            if target.shape != source.shape or target.dtype != source.dtype:
                raise ValueError(
                    f"Runtime weight {update.name!r} mismatch: expected "
                    f"{tuple(target.shape)} {target.dtype}, got {tuple(source.shape)} {source.dtype}"
                )
            # Permit ordinary transposes, but reject broadcast/overlapping views.
            strides = sorted(
                (stride, size) for size, stride in zip(target.shape, target.stride()) if size > 1
            )
            span = 1
            for stride, size in strides:
                if stride < span:
                    raise ValueError(f"Overlapping target view for {update.name!r}")
                span = stride * (size - 1) + span
            if target.numel():
                storage = (target.device, target.untyped_storage().data_ptr())
                if storage in storages:
                    raise ValueError("Aliased targets must be updated through a single owner")
                storages.add(storage)
            for tensor in (target, source):
                if tensor.device.type == "cuda":
                    self._devices.add(tensor.device)

        self._synchronize()
        self._entries = [
            (
                update.name,
                update.target,
                _signature(update.target),
                update.source.detach().to(device="cpu", copy=True),
                update.target.detach().to(device="cpu", copy=True),
            )
            for update in updates
        ]
        self._state = "prepared"

    def _synchronize(self) -> None:
        for device in sorted(self._devices, key=str):
            torch.cuda.synchronize(device)

    def _validate_storage(self) -> None:
        for name, target, signature, _, _ in self._entries:
            if _signature(target) != signature:
                raise RuntimeError(f"Runtime storage changed while updating {name!r}")

    @torch.no_grad()
    def install(self) -> None:
        if self._state != "prepared":
            raise RuntimeError(f"Cannot install a {self._state} weight transaction")
        self._validate_storage()
        self._state = "installing"
        try:
            for _, target, _, source, _ in self._entries:
                target.copy_(source)
            self._synchronize()
        except BaseException:
            self.rollback()
            raise
        self._state = "installed"

    def commit(self) -> None:
        if self._state != "installed":
            raise RuntimeError(f"Cannot commit a {self._state} weight transaction")
        self._validate_storage()
        self._entries.clear()
        self._state = "committed"

    @torch.no_grad()
    def rollback(self) -> None:
        if self._state == "rolled_back":
            return
        if self._state == "committed":
            raise RuntimeError("Cannot roll back committed runtime weights")
        self._validate_storage()
        if self._state != "prepared":
            for _, target, _, _, previous in self._entries:
                target.copy_(previous)
            self._synchronize()
        self._entries.clear()
        self._state = "rolled_back"
