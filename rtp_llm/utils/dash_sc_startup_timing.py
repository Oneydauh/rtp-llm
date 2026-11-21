"""Opt-in ARM test evidence; never record arguments, locals or requests."""

import faulthandler
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

TARGET = "//rtp_llm/test/smoke:deepseek_v41_flash_ced_dspark_pd_cp2_dp2_sm100_arm"
_events = _stacks = None
_pid = None


def mark(stage):
    if _events is not None and _pid == os.getpid():
        print(
            stage,
            time.monotonic_ns(),
            time.process_time_ns(),
            time.thread_time_ns(),
            file=_events,
            flush=True,
        )


@contextmanager
def capture(rank, server, config):
    global _events, _stacks, _pid
    if (
        os.environ.get("RTP_ARM_DASHSC_TIMING") != "1"
        or os.environ.get("TEST_TARGET") != TARGET
    ):
        yield
        return
    service = config.role_config.role_type
    role = str(getattr(service, "name", service)).split(".")[-1].lower()
    if role not in ("prefill", "decode"):
        role = "unknown"
    root = Path(os.environ["TEST_UNDECLARED_OUTPUTS_DIR"]) / "arm_dashsc"
    root.mkdir(exist_ok=True)
    pid = os.getpid()
    name = f"{role}-r{int(rank)}-s{int(server)}-p{pid}"
    with (root / (name + ".events")).open("x") as events, (
        root / (name + ".stack")
    ).open("x") as stacks:
        _events, _stacks, _pid = events, stacks, pid
        print(
            f"main_thread=0x{threading.get_ident():x} native_tid={threading.get_native_id()}",
            file=stacks,
            flush=True,
        )
        mark("start")
        faulthandler.dump_traceback(file=stacks, all_threads=False)
        faulthandler.dump_traceback_later(15, repeat=True, file=stacks)
        try:
            yield
        finally:
            finish("exit")


def finish(stage):
    global _events, _stacks, _pid
    if _events is not None and _pid == os.getpid():
        mark(stage)
        faulthandler.cancel_dump_traceback_later()
        _events = _stacks = _pid = None


def timed(stage, fn, *args, **kwargs):
    if _events is None or _pid != os.getpid():
        return fn(*args, **kwargs)
    mark(stage + ".begin")
    try:
        return fn(*args, **kwargs)
    finally:
        mark(stage + ".end")
