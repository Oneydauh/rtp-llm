import copy
from typing import Any, Callable, Iterator, Optional, Tuple, Type

import torch
from rtp_llm.models_py.read_spec import ReadSpec

ReadSpecResolver = Callable[[str, Tuple[int, ...]], ReadSpec]

_FLOAT8_DTYPES = tuple(
    getattr(torch, name)
    for name in (
        "float8_e4m3fn",
        "float8_e4m3fnuz",
        "float8_e5m2",
        "float8_e5m2fnuz",
        "float8_e8m0fnu",
    )
    if hasattr(torch, name)
)


def _transport_tensor(tensor):
    """Return an FST tensor wrapper with a NCCL-safe view of float8 bytes."""

    raw_tensor = getattr(tensor, "real_tensor", None)
    if (
        not isinstance(raw_tensor, torch.Tensor)
        or raw_tensor.dtype not in _FLOAT8_DTYPES
    ):
        return tensor
    transport = copy.copy(tensor)
    transport.real_tensor = raw_tensor.view(torch.uint8)
    return transport


class _Float8SafeProcessGroup:
    """Preserve float8 payload bits across collectives unsupported by NCCL.

    Some PyTorch/NCCL combinations reject float8 in broadcast, scatter, send,
    and recv. FST's tensor wrappers expose the backing tensor as
    ``real_tensor``; an equal-sized uint8 view makes those collectives portable
    without allocating or converting the payload.
    """

    def __init__(self, process_group):
        self._process_group = process_group

    def __getattr__(self, name):
        return getattr(self._process_group, name)

    def size(self):
        return self._process_group.size()

    def rank(self):
        return self._process_group.rank()

    def broadcast(self, dst, rank):
        return self._process_group.broadcast(_transport_tensor(dst), rank)

    def scatter(self, dst, scatter_list, src):
        return self._process_group.scatter(
            _transport_tensor(dst),
            [_transport_tensor(tensor) for tensor in scatter_list],
            src,
        )

    def send(self, tensor, dst_rank, tag):
        return self._process_group.send(
            _transport_tensor(tensor), dst_rank, tag
        )

    def recv(self, tensor, src_rank, tag):
        return self._process_group.recv(
            _transport_tensor(tensor), src_rank, tag
        )


def _install_float8_safe_process_group(file_buffer):
    if not isinstance(file_buffer.pg, _Float8SafeProcessGroup):
        file_buffer.pg = _Float8SafeProcessGroup(file_buffer.pg)
    return file_buffer.pg


def iter_read_spec_tensors(
    file_buffer,
    keys,
    read_spec_resolver: ReadSpecResolver,
    safetensor_slice_expander: Optional[Any] = None,
) -> Iterator[Tuple[str, object]]:
    """Consume one FST file buffer using its routing/sharding primitives.

    ``get_sharded`` performs a source-side split followed by scatter.  It does
    not make FST's file copier range-aware, but it prevents every non-source
    rank from materializing the full tensor. ``push_tensor`` similarly routes
    a per-expert key directly when it has one owning EP rank. Stacked expert
    tensors use ``get_sharded(dim=0)`` before expansion into per-expert keys.
    """

    pg = _install_float8_safe_process_group(file_buffer)
    pg_size = pg.size()
    pg_rank = pg.rank()
    all_ranks = tuple(range(pg_size))

    for key in keys:
        shape = tuple(file_buffer.get_shape(key))
        if (
            safetensor_slice_expander is not None
            and safetensor_slice_expander.handles_safetensor(key)
        ):
            expanded = safetensor_slice_expander.expand_safetensor(key, shape)
            experts_per_rank = (
                safetensor_slice_expander.num_experts
                // safetensor_slice_expander.ep_size
            )
            expected_start = pg_rank * experts_per_rank
            if (
                safetensor_slice_expander.ep_size != pg_size
                or safetensor_slice_expander.start_expert != expected_start
            ):
                raise RuntimeError(
                    f"FST stacked expert tensor {key!r} requires one WORLD rank "
                    "per EP rank with matching rank order"
                )
            if not hasattr(file_buffer, "get_sharded"):
                raise RuntimeError(
                    "fastsafetensors does not expose get_sharded required for "
                    f"stacked expert tensor {key!r}"
                )
            local_experts = file_buffer.get_sharded(key, 0)
            if not isinstance(local_experts, torch.Tensor):
                raise RuntimeError(f"FST did not return stacked expert tensor {key!r}")
            if local_experts.shape[0] != experts_per_rank:
                raise RuntimeError(
                    f"FST stacked expert shard {key!r} has "
                    f"{local_experts.shape[0]} experts, expected {experts_per_rank}"
                )
            for local_index, (output_name, expert_id) in enumerate(expanded):
                if expert_id != expected_start + local_index:
                    raise RuntimeError(
                        f"Non-contiguous expert expansion for {key!r}: "
                        f"expert {expert_id} at local index {local_index}"
                    )
                yield output_name, local_experts[local_index].contiguous()
            continue

        read_spec = read_spec_resolver(key, shape)
        targets = read_spec.target_ranks
        if targets is not None:
            if any(rank >= pg_size for rank in targets):
                raise ValueError(
                    f"ReadSpec for {key!r} targets ranks {targets}, but FST "
                    f"process group has size {pg_size}"
                )
            should_load = pg_rank in targets
            if read_spec.load != should_load:
                raise RuntimeError(
                    f"ReadSpec for {key!r} is inconsistent on rank {pg_rank}: "
                    f"load={read_spec.load}, target_ranks={targets}"
                )
            if not targets:
                continue
            can_route = hasattr(file_buffer, "push_tensor")
            # FST marks a tensor consumed after ``push_tensor`` and may free
            # its file buffer immediately.  A single targeted push is safe;
            # repeated pushes to TP/DP replicas are not.  Keep the latter on
            # the broadcast fallback until FST exposes multicast/subgroups.
            if len(targets) == 1 and targets != all_ranks and can_route:
                target_rank = targets[0]
                tensor = file_buffer.push_tensor(key, target_rank)
                if pg_rank == target_rank:
                    if not isinstance(tensor, torch.Tensor):
                        raise RuntimeError(
                            f"FST did not return routed tensor {key!r} on "
                            f"target rank {target_rank}"
                        )
                    yield read_spec.output_name, read_spec.materialize(tensor)
                continue
            if targets != all_ranks:
                tensor = file_buffer.get_tensor(key)
                if read_spec.load:
                    yield read_spec.output_name, read_spec.materialize(tensor)
                continue

        if not read_spec.load:
            continue
        can_scatter = (
            read_spec.shard_dim is not None
            and read_spec.shard_count == pg_size
            and read_spec.shard_rank == pg_rank
            and targets in (None, all_ranks)
            and hasattr(file_buffer, "get_sharded")
        )
        if can_scatter:
            tensor = file_buffer.get_sharded(key, read_spec.shard_dim)
            yield read_spec.output_name, read_spec.wrap_sharded_tensor(tensor)
            continue

        tensor = file_buffer.get_tensor(key)
        yield read_spec.output_name, read_spec.materialize(tensor)


def build_read_spec_parallel_loader(
    parallel_loader: Type, timing_context: Type
) -> Type:
    """Build a version-local ParallelLoader subclass without eager FST import."""

    class ReadSpecParallelLoader(parallel_loader):
        def __init__(
            self,
            *args,
            read_spec_resolver: ReadSpecResolver,
            safetensor_slice_expander: Optional[Any] = None,
            **kwargs,
        ):
            super().__init__(*args, **kwargs)
            self.read_spec_resolver = read_spec_resolver
            self.safetensor_slice_expander = safetensor_slice_expander

        def _consume_single_batch(self):
            with timing_context("wait_queue", self._log_message) as timer:
                batch_item = self.batch_queue.get()
                if self.queue_size == 0 and self.consumer_processed is not None:
                    self.consumer_processed.set()
                if batch_item is None:
                    self._log_error("get batch_item is None, will break")
                    return
                if isinstance(batch_item, Exception):
                    self._log_error("get batch_item is Exception, will raise")
                    raise batch_item
                batch = batch_item
                timer.batch_id = batch.batch_id
            queue_wait_time = timer.elapsed_ms

            if queue_wait_time / 1000 > 10:
                self._log_message(
                    f"Warning: Batch {batch.batch_id}: Queue wait took "
                    f"{queue_wait_time:.3f} ms",
                    is_error=True,
                )

            try:
                self._log_message(
                    f"Batch {batch.batch_id}: tensor key len: {len(batch.keys)}"
                )
                with timing_context(
                    "get_tensor", self._log_message, batch.batch_id
                ) as timer:
                    yield from iter_read_spec_tensors(
                        batch.fb,
                        batch.keys,
                        self.read_spec_resolver,
                        self.safetensor_slice_expander,
                    )
                get_tensor_time = timer.elapsed_ms
            finally:
                with timing_context(
                    "fb.close", self._log_message, batch.batch_id
                ) as timer:
                    batch.fb.close()
                close_time = timer.elapsed_ms

            self._log_message(
                f"Batch {batch.batch_id} summary: "
                f"add_filenames={batch.add_filenames_time:.3f}ms, "
                f"copy_files={batch.copy_files_time:.3f}ms, "
                f"get_tensor={get_tensor_time:.3f}ms, "
                f"close={close_time:.3f}ms"
            )
            if self.queue_size < 0 and self.consumer_processed is not None:
                self.consumer_processed.set()

    ReadSpecParallelLoader.__name__ = "ReadSpecParallelLoader"
    return ReadSpecParallelLoader
