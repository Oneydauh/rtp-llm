"""Private opt-in markers for the full V4 xgrammar FST smoke.

Fences are device-local, not collective. A fence observes *all* work submitted
on this device so overlapping producer/consumer batches are candidates, not
proof of the identity of a failing kernel.
"""

import ctypes
import logging
import os
import re
import time
from pathlib import Path

import torch

log = logging.getLogger(__name__)


def _pci(ordinal):
    # CUDA ordinals respect the prefill role's CUDA_VISIBLE_DEVICES remapping.
    # Do not use the NVML index as if it were this process's CUDA ordinal.
    try:
        driver = ctypes.CDLL("libcuda.so.1")
        dev = ctypes.c_int()
        bus = ctypes.create_string_buffer(32)
        if driver.cuDeviceGet(ctypes.byref(dev), ordinal) == 0:
            if driver.cuDeviceGetPCIBusId(bus, len(bus), dev) == 0:
                return bus.value.decode("ascii")
    except (AttributeError, OSError, UnicodeError):
        pass
    return "unknown"


def _namespace_pids():
    # If procfs is container-local Linux need not expose the host PID. Never
    # label an inner namespace PID as the host PID.
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("NSpid:"):
                return (
                    "/".join(v for v in line.split()[1:] if v.isdecimal()) or "unknown"
                )
    except OSError:
        pass
    return "unknown"


class V4FstTrace:
    def __init__(self, mode, rank, device):
        self.mode, self.rank, self.device = mode, rank, device
        self._batches = {}

    @classmethod
    def from_env(cls, device):
        mode = os.environ.get("RTP_V4_XGRAMMAR_FST_DIAG")
        if mode not in ("trace", "fenced"):
            raise ValueError("Expected trace or fenced for V4 FST diagnostics")
        device = torch.device(device)
        ordinal = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        device = torch.device("cuda", ordinal)
        trace = cls(mode, torch.distributed.get_rank(), device)
        log.info(
            "V4FST mode=%s role=prefill rank=%s pid=%d nspid=%s cuda=%d pci=%s monotonic=%.6f",
            mode,
            trace.rank,
            os.getpid(),
            _namespace_pids(),
            ordinal,
            _pci(ordinal),
            time.monotonic(),
        )
        return trace

    def point(self, stage, batch=None, *, layer=None, fence=True):
        # On long successful startups, only report the first attention layer
        # and an actual failed fence, not per-layer success hundreds of times.
        emit = layer is None or layer == 0
        fields = (
            self.mode,
            self.rank,
            stage,
            batch,
            self._batches.get(batch, "-"),
            layer,
        )
        fmt = "V4FST mode=%s rank=%s stage=%s batch=%s shards=%s layer=%s"
        if emit:
            log.info(fmt + " marker=host", *fields)
        if self.mode == "fenced" and fence:
            try:
                torch.cuda.synchronize(self.device)
            except Exception:
                log.error(fmt + " marker=fence_failed", *fields)
                raise
            if emit:
                log.info(fmt + " marker=fence_ok", *fields)

    def reader_class(self, base, files):
        index = {os.fspath(path): idx for idx, path in enumerate(sorted(files), 1)}
        trace = self

        def shard_id(path):
            # Emit only a numerical checkpoint ordinal, never paths, tensor
            # names, metadata, or the contents of the checkpoint.
            m = re.search(r"-(\d+)-of-(\d+)\.safetensors$", os.path.basename(path))
            return (
                f"{int(m[1])}/{int(m[2])}"
                if m
                else f"selected:{index.get(os.fspath(path), 0)}"
            )

        class TracedReader(base):
            _rtp_v4_fst_trace = trace

            def _load_single_batch(self, batch_id, file_list):
                trace._batches[batch_id] = ",".join(shard_id(p) for p in file_list)
                # Producer thread: no device sync, no touching its CUDA context
                # or changing when the native producer may enqueue the batch.
                trace.point("producer_start", batch_id, fence=False)
                result = super()._load_single_batch(batch_id, file_list)
                trace.point("producer_returned", batch_id, fence=False)
                return result

        return TracedReader
