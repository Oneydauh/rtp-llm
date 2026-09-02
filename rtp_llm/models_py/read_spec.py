from dataclasses import dataclass, replace
from typing import Any, Optional, Tuple, Union

import torch

ReadIndex = Union[int, slice]


@dataclass(frozen=True)
class ReadSpec:
    """Describe how one checkpoint tensor is consumed by the current rank.

    ``slices`` is expressed in the checkpoint tensor's original layout.  A
    backend may execute it while reading (safetensors), while distributing
    (fastsafetensors ``get_sharded``), or as a compatibility fallback after a
    full tensor has been materialized.
    """

    source_name: str
    output_name: str
    source_shape: Tuple[int, ...]
    load: bool = True
    slices: Optional[Tuple[ReadIndex, ...]] = None
    partition: str = "replicated"
    shard_dim: Optional[int] = None
    shard_count: int = 1
    shard_rank: int = 0
    target_ranks: Optional[Tuple[int, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_name, str) or not self.source_name:
            raise ValueError("ReadSpec.source_name must be a non-empty string")
        if not isinstance(self.output_name, str) or not self.output_name:
            raise ValueError("ReadSpec.output_name must be a non-empty string")
        if not isinstance(self.source_shape, tuple) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 0
            for size in self.source_shape
        ):
            raise ValueError(f"Invalid ReadSpec source shape {self.source_shape!r}")
        if not isinstance(self.load, bool):
            raise TypeError("ReadSpec.load must be a bool")
        if self.slices is not None:
            if not isinstance(self.slices, tuple):
                raise TypeError("ReadSpec.slices must be a tuple or None")
            if len(self.slices) != len(self.source_shape):
                raise ValueError(
                    f"ReadSpec for {self.source_name!r} has {len(self.slices)} "
                    f"indices for a {len(self.source_shape)}-D tensor"
                )
            if any(not isinstance(index, (int, slice)) for index in self.slices):
                raise TypeError("ReadSpec indices must be int or slice")
        if not isinstance(self.partition, str) or not self.partition:
            raise ValueError("ReadSpec.partition must be a non-empty string")
        if (
            isinstance(self.shard_count, bool)
            or not isinstance(self.shard_count, int)
            or self.shard_count <= 0
        ):
            raise ValueError("ReadSpec.shard_count must be a positive integer")
        if (
            isinstance(self.shard_rank, bool)
            or not isinstance(self.shard_rank, int)
            or not 0 <= self.shard_rank < self.shard_count
        ):
            raise ValueError(
                f"Invalid ReadSpec shard rank {self.shard_rank}/" f"{self.shard_count}"
            )
        if self.shard_dim is not None:
            if not self.source_shape:
                raise ValueError("Scalar ReadSpec cannot have shard_dim")
            if not 0 <= self.shard_dim < len(self.source_shape):
                raise ValueError(
                    f"ReadSpec shard_dim={self.shard_dim} is invalid for "
                    f"shape {self.source_shape}"
                )
            if self.slices is None:
                raise ValueError("ReadSpec with shard_dim must define slices")
        if self.target_ranks is not None:
            if not isinstance(self.target_ranks, tuple):
                raise TypeError("ReadSpec.target_ranks must be a tuple or None")
            if any(
                isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
                for rank in self.target_ranks
            ):
                raise ValueError(f"Invalid ReadSpec target ranks {self.target_ranks!r}")
            if tuple(sorted(set(self.target_ranks))) != self.target_ranks:
                raise ValueError("ReadSpec.target_ranks must be sorted and unique")

    @classmethod
    def full(cls, name: str, shape: Tuple[int, ...]) -> "ReadSpec":
        return cls(name, name, tuple(shape))

    @classmethod
    def skip(cls, name: str, shape: Tuple[int, ...]) -> "ReadSpec":
        return cls(name, name, tuple(shape), load=False, target_ranks=())

    @property
    def is_local_shard(self) -> bool:
        return self.slices is not None

    def for_checkpoint_name(self, name: str) -> "ReadSpec":
        return replace(self, source_name=name, output_name=name)

    def with_targets(self, target_ranks: Tuple[int, ...], load: bool) -> "ReadSpec":
        return replace(self, target_ranks=target_ranks, load=load)

    def materialize(self, source: Any):
        if not self.load:
            raise RuntimeError(
                f"Cannot materialize skipped checkpoint tensor {self.source_name!r}"
            )
        tensor = source if self.slices is None else source[self.slices]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Read backend returned {type(tensor).__name__} for "
                f"{self.source_name!r}, expected torch.Tensor"
            )
        if self.slices is None:
            return tensor
        return LoadedCheckpointTensor(tensor.contiguous(), self)

    def wrap_sharded_tensor(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Read backend returned {type(tensor).__name__} for "
                f"{self.source_name!r}, expected torch.Tensor"
            )
        if self.slices is None:
            return tensor
        return LoadedCheckpointTensor(tensor.contiguous(), self)


@dataclass(frozen=True)
class LoadedCheckpointTensor:
    """Tensor plus proof that the backend already applied ``ReadSpec``."""

    tensor: torch.Tensor
    read_spec: ReadSpec

    def __post_init__(self) -> None:
        if not isinstance(self.tensor, torch.Tensor):
            raise TypeError("LoadedCheckpointTensor.tensor must be torch.Tensor")
        if not self.read_spec.is_local_shard:
            raise ValueError(
                "LoadedCheckpointTensor requires a ReadSpec with a local slice"
            )


def unwrap_checkpoint_tensor(value: Any) -> Tuple[torch.Tensor, Optional[ReadSpec]]:
    if isinstance(value, LoadedCheckpointTensor):
        return value.tensor, value.read_spec
    if not isinstance(value, torch.Tensor):
        raise TypeError(
            f"Checkpoint value must be torch.Tensor or LoadedCheckpointTensor, "
            f"got {type(value).__name__}"
        )
    return value, None


def make_sharded_read_spec(
    name: str,
    shape: Tuple[int, ...],
    dim: int,
    shard_count: int,
    shard_rank: int,
    partition: str = "tp",
) -> ReadSpec:
    shape = tuple(shape)
    if not shape:
        raise ValueError(f"Cannot shard scalar checkpoint tensor {name!r}")
    normalized_dim = dim if dim >= 0 else len(shape) + dim
    if not 0 <= normalized_dim < len(shape):
        raise ValueError(f"Invalid shard dim {dim} for {name!r} shape {shape}")
    if shard_count <= 0 or not 0 <= shard_rank < shard_count:
        raise ValueError(f"Invalid shard rank {shard_rank}/{shard_count}")
    if shape[normalized_dim] % shard_count:
        raise ValueError(
            f"Checkpoint tensor {name!r} dimension {normalized_dim}="
            f"{shape[normalized_dim]} is not divisible by {shard_count}"
        )
    shard_size = shape[normalized_dim] // shard_count
    start = shard_rank * shard_size
    indices = [slice(None)] * len(shape)
    indices[normalized_dim] = slice(start, start + shard_size)
    return ReadSpec(
        source_name=name,
        output_name=name,
        source_shape=shape,
        slices=tuple(indices),
        partition=partition,
        shard_dim=normalized_dim,
        shard_count=shard_count,
        shard_rank=shard_rank,
    )
