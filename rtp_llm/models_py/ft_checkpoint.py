"""Boundary reader for the rank-local format emitted by weights_convert.

FT checkpoints contain *runtime* tensors, not HF weights. This reader never
routes them through HF sharding or quantization post-processing. A model must
explicitly declare each supported boundary conversion before using it.
"""

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

import torch
from safetensors import safe_open

_KEY = re.compile(r"^rank_(\d+)_(\d+)_(\d+)\.(global\..+|layers\.\d+\..+)$")


def validate_format_parameter_ownership(model: torch.nn.Module) -> None:
    """Reject loans before format adapters can write or migrate their storage.

    Existing FT/Meta adapters import complete models, not draft checkpoints
    whose embedding/head are borrowed from a live target. Supporting that
    combination needs an explicit source-exclusion and postprocess contract;
    silently copying into the borrowed Parameter would corrupt the target.
    """
    borrowed = [
        f"{path}.{name}" if path else name
        for path, module in model.named_modules()
        for name in getattr(module, "_rtp_borrowed_parameter_names", ())
    ]
    if borrowed:
        raise PermissionError(
            "FT/Meta checkpoint adapters cannot overwrite borrowed parameters: "
            + ", ".join(sorted(borrowed))
        )


def _unique_json_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError(f"Duplicate FT index key: {name!r}")
        result[name] = value
    return result


@dataclass(frozen=True)
class CheckpointTopology:
    tp_size: int = 1
    dp_size: int = 1
    ep_size: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    ep_rank: int = 0

    def __post_init__(self):
        for axis in ("tp", "dp", "ep"):
            size, rank = getattr(self, axis + "_size"), getattr(self, axis + "_rank")
            if type(size) is not int or size < 1:
                raise ValueError(f"{axis}_size must be a positive integer")
            if type(rank) is not int or not 0 <= rank < size:
                raise ValueError(f"{axis}_rank={rank!r} is outside size={size}")


@dataclass(frozen=True)
class RuntimeTensorBinding:
    """An ephemeral import conversion, owned by the model's format adapter.

    Names in ``sources`` are the historical index suffixes (global./layers.).
    ``target`` names an actual model parameter/buffer, not another weight map.
    Converters must be pure and return the destination layout, already local.
    """

    sources: tuple[str, ...]
    target: str
    convert: Callable[[tuple[torch.Tensor, ...]], torch.Tensor]
    read: Callable[[], tuple[torch.Tensor, ...]] | None = None
    install: Callable[[tuple[torch.Tensor, ...]], None] | None = None


class FTCheckpoint:
    def __init__(
        self,
        root: Path,
        topology: CheckpointTopology,
        entries: Mapping[str, tuple[str, Path]],
    ):
        self.root = root
        self.topology = topology
        self.entries = dict(entries)
        self.read_backend = "scratch"
        self._overlay = {}

    def apply_overlay(self, overlay):
        # Preserve the database's exact checkpoint-name override semantics.
        # Keep mmap-backed references, never a second runtime weight namespace.
        self._overlay = dict(overlay.weights())

    @classmethod
    def open(cls, model_path: str, topology: CheckpointTopology):
        root = Path(model_path).resolve()
        index = root / "model.safetensors.index.json"
        if not index.is_file():
            return None
        with index.open(encoding="utf-8") as stream:
            metadata = json.load(stream, object_pairs_hook=_unique_json_object)
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint index must be an object")
        if metadata.get("is_ft_style_weight") is not True:
            return None
        env = metadata.get("__env__params__", {})
        if not isinstance(env, dict):
            raise ValueError("FT __env__params__ must be an object")
        if str(env.get("PP_SIZE", "1")) != "1":
            raise ValueError(
                "FT pipeline stage-local checkpoint import is not supported"
            )
        # Match legacy default semantics when older exports omit these fields.
        for axis in ("tp", "dp", "ep"):
            expected = getattr(topology, axis + "_size")
            raw = env.get(axis.upper() + "_SIZE", expected)
            if isinstance(raw, bool) or not re.fullmatch(r"[1-9]\d*", str(raw)):
                raise ValueError(f"Invalid FT {axis.upper()}_SIZE: {raw!r}")
            if int(raw) != expected:
                raise ValueError(
                    f"FT {axis.upper()} topology mismatch: checkpoint={raw}, requested={expected}"
                )
        weight_map = metadata.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("FT checkpoint requires a nonempty weight_map")
        requested = (topology.tp_rank, topology.dp_rank, topology.ep_rank)
        sizes = (topology.tp_size, topology.dp_size, topology.ep_size)
        entries = {}
        for key, filename in weight_map.items():
            match = _KEY.fullmatch(key)
            if match is None:
                raise ValueError(f"Malformed FT runtime tensor key: {key!r}")
            ranks = tuple(int(match[i]) for i in (1, 2, 3))
            if any(rank >= size for rank, size in zip(ranks, sizes)):
                raise ValueError(f"FT tensor rank outside requested topology: {key}")
            if not isinstance(filename, str) or not filename:
                raise ValueError(f"Invalid FT shard for {key}")
            path = (root / filename).resolve()
            if Path(filename).is_absolute() or root not in path.parents:
                raise ValueError(f"FT shard escapes checkpoint directory: {filename!r}")
            if path.suffix != ".safetensors" or not path.is_file():
                raise ValueError(f"Missing or unsupported FT shard: {filename!r}")
            if ranks == requested:
                suffix = match[4]
                if suffix in entries:
                    raise ValueError(f"Duplicate logical FT tensor: {suffix}")
                entries[suffix] = (key, path)
        if not entries:
            raise ValueError(f"FT checkpoint has no tensors for rank {requested}")
        checkpoint = cls(root, topology, entries)
        checkpoint.env_params = env
        return checkpoint

    def read(self, name: str) -> torch.Tensor:
        key, path = self.entries[name]
        return self._read_tensor(key, path)

    def _read_tensor(self, key, path, slices=None):
        if key in getattr(self, "_overlay", {}):
            tensor = self._overlay[key]
            return tensor if slices is None else tensor[slices]
        if self.read_backend == "fastsafetensors":
            from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
            from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor

            def resolve(name, shape):
                return ReadSpec(
                    source_name=name,
                    output_name=name,
                    source_shape=tuple(shape),
                    slices=slices,
                )

            values = list(
                iter_fst_range_tensors(
                    [str(path)],
                    name_filter=lambda name: name == key,
                    read_spec_resolver=resolve,
                )
            )
            if len(values) != 1 or values[0][0] != key:
                raise ValueError(f"FT index tensor {key!r} missing from {path.name}")
            # The FST reader has already applied the requested local slice.
            return unwrap_checkpoint_tensor(values[0][1])[0]
        if self.read_backend != "scratch":
            raise ValueError(f"Unknown FT read backend: {self.read_backend}")
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            if key not in shard.keys():
                raise ValueError(f"FT index tensor {key!r} missing from {path.name}")
            return (
                shard.get_tensor(key)
                if slices is None
                else shard.get_slice(key)[slices]
            )

    def read_rank_slice(self, name: str, tp_rank: int, slices=None):
        """Read a bounded slice from a peer TP file without a collective."""
        peer = (
            self
            if tp_rank == self.topology.tp_rank
            else self.open(str(self.root), replace(self.topology, tp_rank=tp_rank))
        )
        key, path = peer.entries[name]
        if slices is not None:
            return self._read_tensor(key, path, slices)
        if key in self._overlay:
            return tuple(self._overlay[key].shape)
        with safe_open(str(path), framework="pt", device="cpu") as shard:
            view = shard.get_slice(key)
            return tuple(view.get_shape())

    @torch.no_grad()
    def load_parameters(
        self,
        model: torch.nn.Module,
        bindings: tuple[RuntimeTensorBinding, ...],
        source_validators: Mapping[str, Callable[[torch.Tensor], None]] | None = None,
    ) -> None:
        """Validate the entire declared layout, then copy into existing storage.

        This is deliberately not a generic model loading entry point. The model
        adapter must prove quantization/device layouts and declare all bindings.
        Unknown source tensors or unbound destination parameters are errors.
        """
        validate_format_parameter_ownership(model)
        targets = dict(model.named_parameters(remove_duplicate=False))
        for path, module in model.named_modules():
            for name, value in module._buffers.items():
                if value is not None and name not in module._non_persistent_buffers_set:
                    targets[f"{path}.{name}" if path else name] = value
        used_sources, used_targets = set(), set()
        for name, validate in (source_validators or {}).items():
            validate(self.read(name))
            used_sources.add(name)
        target_owners = {}
        for binding in bindings:
            if not binding.sources:
                raise ValueError(f"FT binding has no source: {binding.target}")
            if binding.target in used_targets or binding.target not in targets:
                raise ValueError(f"Duplicate or unknown FT target: {binding.target}")
            if any(name not in self.entries for name in binding.sources):
                raise ValueError(
                    f"Missing FT sources for {binding.target}: {binding.sources}"
                )
            values = (
                binding.read()
                if binding.read
                else tuple(self.read(name) for name in binding.sources)
            )
            tensor = binding.convert(values)
            target = targets[binding.target]
            if id(target) in target_owners:
                raise ValueError(
                    "FT tied parameter bindings require an explicit model adapter: "
                    f"{target_owners[id(target)]}, {binding.target}"
                )
            target_owners[id(target)] = binding.target
            if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape:
                raise ValueError(f"FT layout mismatch for {binding.target}")
            # Runtime snapshots must not be silently requantized/cast.
            if tensor.dtype != target.dtype:
                raise ValueError(
                    f"FT dtype mismatch for {binding.target}: {tensor.dtype} vs {target.dtype}"
                )
            used_sources.update(binding.sources)
            used_targets.add(binding.target)
            del tensor
        if used_sources != self.entries.keys():
            raise ValueError(
                f"Unbound FT tensors: {sorted(self.entries.keys() - used_sources)}"
            )
        if used_targets != targets.keys():
            raise ValueError(
                f"Unbound model tensors: {sorted(targets.keys() - used_targets)}"
            )
        # Validate before mutation, but do not retain a model-sized second set of
        # runtime weights. This conservative boundary implementation reads twice.
        # Exported checkpoint files must remain immutable throughout loading.
        for binding in bindings:
            path = binding.target
            target = targets[path]
            values = (
                binding.read()
                if binding.read
                else tuple(self.read(name) for name in binding.sources)
            )
            tensor = binding.convert(values)
            if binding.install is None:
                target.copy_(tensor)
            else:
                # Module-owned loaders also own checkpoint provenance flags and
                # tied-parameter fan-out. Reuse them after all layout validation.
                binding.install(values)
            owner_name, _, name = path.rpartition(".")
            owner = model.get_submodule(owner_name) if owner_name else model
            marker = getattr(owner, "_record_parameter_loaded", None)
            if marker is None:
                marker = getattr(owner, "_mark_weight_loaded", None)
            if marker is not None:
                marker(name)
