"""Keep V4.1 FST file-sized staging allocations out of the model/KV allocator."""

import torch


def v41_file_staging_loader(base, model_type, device, *, cuda_backend, use_shm):
    if (
        model_type not in ("deepseek_v41", "deepseek_v41_dspark")
        or device.type != "cuda"
        or cuda_backend != "Cuda"
        or use_shm
        or not callable(getattr(base, "_load_single_batch", None))
        or not callable(getattr(torch.cuda, "MemPool", None))
        or not callable(getattr(torch.cuda, "use_mem_pool", None))
    ):
        return base
    get_backend = getattr(torch.cuda, "get_allocator_backend", None)
    if not callable(get_backend):
        return base
    try:
        if get_backend() != "native":
            return base
    except (AttributeError, RuntimeError):
        # This is a staging optimization, not an extra FST prerequisite.
        return base

    class StagedFileLoader(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Allocate and bind the pool to the destination, not the process's
            # default GPU. It lives as long as this reader (including its producer).
            with torch.cuda.device(device):
                self._rtp_file_pool = torch.cuda.MemPool()

        def _load_single_batch(self, *args, **kwargs):
            # FST allocates its raw shard buffers using caching_allocator_alloc
            # in the producer thread. Setting a pool around iterate_weights()
            # on the consumer would only capture model-side copies/parameters.
            with torch.cuda.device(device), torch.cuda.use_mem_pool(
                self._rtp_file_pool, device
            ):
                return super()._load_single_batch(*args, **kwargs)

    return StagedFileLoader
