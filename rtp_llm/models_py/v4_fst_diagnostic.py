"""Private opt-in markers for the full V4 xgrammar FST smoke.

Fences are device-local, not collective. A fence observes *all* work submitted
on this device so overlapping producer/consumer batches are candidates, not
proof of the identity of a failing kernel.
"""

import ctypes
import hashlib
import logging
import os
import re
import time
from contextlib import contextmanager
from pathlib import Path

import torch

log = logging.getLogger(__name__)


class V4RouterRangeError(ValueError):
    """This table failed the existing model gate ABI validation, not a retry."""

    def __init__(self, layer, num_experts, observed, lo, hi):
        self.layer, self.num_experts = layer, num_experts
        self.observed, self.lo, self.hi = observed, lo, hi
        super().__init__(
            f"gate.tid2eid contains an out-of-range expert index: layer={layer} "
            f"dtype={observed.dtype} shape={tuple(observed.shape)} "
            f"observed=[{lo},{hi}] valid=[0,{num_experts})"
        )


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
    def compare_bad_router(self, error, catalog_fn, selected_files):
        """Compare at most 8 MiB on the failed worker; never use it for loading."""
        if not isinstance(error, V4RouterRangeError):
            return
        try:
            observed = error.observed
            if (
                type(error.layer) is not int
                or not 0 <= error.layer < 128
                or observed.dim() != 2
                or observed.numel() == 0
                or observed.numel() * observed.element_size() > 8 * 1024**2
                or observed.dtype not in (torch.int32, torch.int64)
            ):
                raise ValueError("unsupported table")
            # A validated weight index from this loader, not a path fabricated
            # from a tensor name, message, environment variable or exception.
            name = f"layers.{error.layer}.ffn.gate.tid2eid"
            files = {os.path.realpath(p) for p in selected_files}
            sources = [
                p
                for k, p in catalog_fn().entries
                if k == name and os.path.realpath(p) in files
            ]
            if len(sources) != 1:
                raise ValueError("ambiguous source")
            from safetensors import safe_open

            with safe_open(sources[0], framework="pt", device="cpu") as reader:
                metadata = reader.get_slice(name)
                expected = "I32" if observed.dtype == torch.int32 else "I64"
                if (
                    tuple(metadata.get_shape()) != tuple(observed.shape)
                    or metadata.get_dtype() != expected
                ):
                    raise ValueError("source metadata mismatch")
                reference = reader.get_tensor(name)
                actual = observed.detach().to("cpu").contiguous()

                def digest(value):
                    return hashlib.sha256(
                        memoryview(value.view(torch.uint8).numpy())
                    ).hexdigest()

                log.error(
                    "V4FST failed_router rank=%s layer=%d dtype=%s shape=%s "
                    "observed_range=[%d,%d] source_range=[%d,%d] "
                    "equal=%s observed_sha256=%s source_sha256=%s",
                    self.rank,
                    error.layer,
                    observed.dtype,
                    tuple(observed.shape),
                    error.lo,
                    error.hi,
                    reference.min().item(),
                    reference.max().item(),
                    torch.equal(actual, reference),
                    digest(actual),
                    digest(reference),
                )
        except Exception as diagnostic_error:
            # Even a poisoned CUDA context or inaccessible file must preserve
            # the *original* validation rejection. Exception strings can leak paths.
            log.error(
                "V4FST failed_router unavailable rank=%s layer=%s reason=%s",
                self.rank,
                error.layer,
                type(diagnostic_error).__name__,
            )

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

    @contextmanager
    def watch_keys(self, batch, keys):
        """Identify native-read/consumer progress without recording checkpoint keys."""
        ordinal = None

        def indexed():
            nonlocal ordinal
            report_after = 0.0
            for ordinal, key in enumerate(keys):
                now = time.monotonic()
                if now >= report_after:
                    log.info(
                        "V4FST consumer_progress rank=%s batch=%s key_ordinal=%s",
                        self.rank, batch, ordinal,
                    )
                    report_after = now + 0.5
                yield key

        try:
            yield indexed()
        except (Exception, GeneratorExit) as exc:
            # GeneratorExit also occurs when external module dispatch fails (or
            # stops intentionally). This ordinal is a location, not a verdict.
            log.warning(
                "V4FST consumer_stopped rank=%s batch=%s key_ordinal=%s reason=%s",
                self.rank, batch, ordinal, type(exc).__name__,
            )
            raise

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
                try:
                    result = super()._load_single_batch(batch_id, file_list)
                except Exception:
                    trace.point("producer_raised", batch_id, fence=False)
                    raise
                trace.point("producer_returned", batch_id, fence=False)
                return result

        return TracedReader
