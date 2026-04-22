#!/usr/bin/env python3
"""PD-separated perf + torch profiler driver.

Points at already-deployed prefill/decode workers. For each shape point it
drives warmup → measure → (optional) profile pass. Profiling is armed via
the per-worker ``/start_profile`` endpoint (global one-shot window), not
per-request ``gen_timeline=true``.

One seed URL per role is enough — rtp-llm's ``/update_scheduler_info`` and
``/start_profile`` broadcast internally to every DP/TP rank.

Usage:
  # prefill-only sweep + capture
  python -m rtp_llm.test.perf_test.remote_pd_profile \\
      --prefill-url http://prefill_worker:22530 \\
      --seq-lens 128,512,2048,8192 --capture-profile

  # decode-only sweep + capture (direct traffic — works only for co-located
  # deployments; PD-separated decode workers reject raw '/', see below)
  python -m rtp_llm.test.perf_test.remote_pd_profile \\
      --decode-url http://decode_worker:22530 \\
      --batch-sizes 16,32,64 --input-len 512 --decode-length 128 \\
      --capture-profile

  # full PD pipeline in one shot — decode traffic is routed THROUGH the
  # prefill URL (needed for PD-sep handoff); profile is armed on the decode
  # worker. Traces land on whichever worker is the profile target.
  python -m rtp_llm.test.perf_test.remote_pd_profile \\
      --prefill-url http://prefill:22530 --decode-url http://decode:22530 \\
      --seq-lens 128,512,2048,8192 \\
      --batch-sizes 16,32,64 --input-len 512 --decode-length 128 \\
      --capture-profile --profile-trace-name daily_probe

Timeline JSONs land on each worker's ``$TORCH_CUDA_PROFILER_DIR`` with
prefix ``<profile_trace_name>_<YYYYMMDD_HHMMSS>_{prefill_seq,decode_bs}<N>_wr{rank}_*.json``.
Collect from workers yourself (rsync/scp) — this driver only triggers capture.

Notes:
  * Each URL must be a direct worker (the frontend on a role), not a gateway.
  * PD-separated decode workers don't implement ``GenerateStreamCall`` on the
    raw ``/`` endpoint — they only consume prefill→decode KV handoffs. If you
    pass both URLs, decode sweep traffic is driven through the prefill URL.
  * Server must have been started with ``--torch_cuda_profiler_dir <dir>`` or
    ``TORCH_CUDA_PROFILER_DIR`` env for ``--capture-profile`` to produce files.
  * The prefill frontend caps in-flight with ``CONCURRENCY_LIMIT``; overflow
    returns HTTP 500 + ``error_code=409``. We retry-with-backoff so batches
    drip through.
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter

log = logging.getLogger("remote_pd_profile")


def _new_session(auth_headers: Dict[str, str], pool_size: int = 256) -> requests.Session:
    """Session with a generous connection pool so big fan-outs don't spam
    'Connection pool is full' warnings from urllib3.
    """
    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update(auth_headers)
    return s


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _approx_prompt(input_len_tokens: int) -> str:
    return "hello " * max(1, input_len_tokens)


def _tokenizer_prompt(tokenizer, input_len: int) -> str:
    base = "hello " * (input_len + 20)
    left, right = 0, len(base)
    while left < right:
        mid = (left + right) // 2
        n = len(tokenizer.encode(base[:mid]))
        if n == input_len:
            return base[:mid]
        if n < input_len:
            left = mid + 1
        else:
            right = mid
    return base[:left]


def _load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer  # type: ignore

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _build_body(
    prompt: str,
    is_decode: bool,
    decode_length: int,
    generate_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Raw generate body. Profiling is triggered externally via /start_profile;
    no ``gen_timeline`` fields are stuffed into the body.
    """
    n_new = decode_length if is_decode else 1
    gen_cfg: Dict[str, Any] = {
        "max_new_tokens": n_new,
        "min_new_tokens": n_new,
        "force_sp_accept": True,
    }
    if generate_config:
        gen_cfg.update(generate_config)

    return {
        "prompt": prompt,
        "generate_config": gen_cfg,
        "top_k": generate_config.get("top_k", 1) if generate_config else 1,
    }


def _start_profile(
    session: requests.Session,
    base_url: str,
    trace_name: str,
    num_steps: int,
    start_step: int = 1,
    enable_all_rank: bool = False,
) -> None:
    """Arm a one-shot profile window on the target worker. Payload matches
    ``LocalRpcServer.cc:start_profile``; with ``enable_all_rank=true`` the
    TP rank-0 frontend broadcasts to every rank in the group.

    Defaults:
      * ``start_step=1`` — skip the first engine step after arming. The
        ``/start_profile`` HTTP reply returns before the configure actually
        reaches every rank via gRPC, so the very first step can race the
        config and produce an empty trace. One step of buffer avoids that.
      * ``enable_all_rank=false`` — only the frontend's TP rank (rank 0)
        captures. Full-rank capture (all TP×DP) produces huge files that
        aren't worth the load when TP ranks run symmetric kernels anyway.
    """
    url = base_url.rstrip("/") + "/start_profile"
    payload = {
        "trace_name": trace_name,
        "start_step": start_step,
        "num_steps": num_steps,
        "enable_all_rank": enable_all_rank,
    }
    resp = session.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            "start_profile failed: status=%d body=%s"
            % (resp.status_code, resp.text)
        )
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if data.get("status", "ok") != "ok" or "error" in data:
        raise RuntimeError("start_profile rejected: %s" % resp.text)
    log.info(
        "  profile armed: trace=%s start_step=%d num_steps=%d all_rank=%s",
        trace_name, start_step, num_steps, enable_all_rank,
    )


def _pin_scheduler(
    session: requests.Session, base_url: str, local_batch: int, mode: str,
) -> None:
    url = base_url.rstrip("/") + "/update_scheduler_info"
    payload = {"batch_size": local_batch, "mode": mode}
    resp = session.post(url, json=payload, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(
            "update_scheduler_info failed: status=%d body=%s"
            % (resp.status_code, resp.text)
        )
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if data.get("status", "ok") != "ok":
        raise RuntimeError("scheduler rejected: %s" % resp.text)
    log.info("  scheduler pinned: local_batch=%d mode=%s", local_batch, mode)


def _topology_tag(status: Dict[str, Any], role_letter: str) -> str:
    """Format '<role>_<tp>TP<dp>DP' from a /worker_status payload — e.g.
    'P_8TP1DP', 'D_1TP32DP'. Used to auto-name profile trace files by
    cluster shape so multiple concurrent sweeps don't collide and a
    filename alone identifies which deployment it came from.
    """
    tp = int(status.get("tp_size", 1) or 1)
    dp = int(status.get("dp_size", 1) or 1)
    return "%s_%dTP%dDP" % (role_letter, tp, dp)


def _probe_topology_prefix(
    auth_headers: Dict[str, str],
    prefill_url: Optional[str],
    decode_urls: Optional[List[str]],
) -> str:
    """Probe worker_status on the prefill seed + first decode seed and
    build a compound tag like 'P_8TP1DP-D_1TP32DP'. Returns '' if both
    probes fail (caller falls back to a plain tag).
    """
    parts: List[str] = []
    s = _new_session(auth_headers)
    try:
        if prefill_url:
            try:
                st = _fetch_worker_status(s, prefill_url)
                parts.append(_topology_tag(st, "P"))
            except Exception as e:
                log.warning(
                    "topology probe on prefill %s failed: %s",
                    prefill_url, e,
                )
        if decode_urls:
            try:
                st = _fetch_worker_status(s, decode_urls[0])
                parts.append(_topology_tag(st, "D"))
            except Exception as e:
                log.warning(
                    "topology probe on decode %s failed: %s",
                    decode_urls[0], e,
                )
    finally:
        s.close()
    return "-".join(parts)


def _role(status: Dict[str, Any]) -> str:
    """Server returns role as RoleType enum str repr ('RoleType.PREFILL').
    Strip the prefix and lowercase.
    """
    s = str(status.get("role") or "?").lower()
    return s.split(".", 1)[1] if s.startswith("roletype.") else s


def _fetch_worker_status(
    session: requests.Session, base_url: str,
) -> Dict[str, Any]:
    """POST /worker_status and return role/dp_size/tp_size.

    /update_scheduler_info broadcasts internally to all DP addrs
    (``rtp_llm/utils/grpc_client_wrapper.py: update_scheduler_info``), so
    one seed URL per role is enough.
    """
    url = base_url.rstrip("/") + "/worker_status"
    resp = session.post(url, json={}, timeout=10)
    if resp.status_code != 200:
        raise RuntimeError(
            "worker_status failed: status=%d body=%s"
            % (resp.status_code, resp.text[:500])
        )
    data = resp.json()
    if "error" in data:
        raise RuntimeError("worker_status error: %s" % data["error"])
    info = {
        "role": data.get("role", "?"),
        "dp_size": int(data.get("dp_size", 1) or 1),
        "tp_size": int(data.get("tp_size", 1) or 1),
        "alive": bool(data.get("alive", True)),
    }
    log.info(
        "  worker_status[%s]: role=%s dp_size=%d tp_size=%d alive=%s",
        base_url, info["role"], info["dp_size"], info["tp_size"], info["alive"],
    )
    return info


def _one_request(
    session: requests.Session, url: str, body: Dict[str, Any], timeout: int,
    max_retries: int = 20, backoff: float = 0.3,
) -> Dict[str, Any]:
    """One generate request with retries on 409 CONCURRENCY_LIMIT.

    rtp-llm's prefill frontend caps in-flight requests; when exceeded it
    returns HTTP 500 with ``error_code=409``. Back off and retry so the
    batch drips through instead of failing hard.
    """
    t0 = time.monotonic()
    attempts = 0
    last_status = None
    last_body: Any = None
    while True:
        try:
            resp = session.post(url, json=body, timeout=timeout)
            try:
                data = resp.json()
            except ValueError:
                data = {}
            last_status = resp.status_code
            last_body = data
            err_code = data.get("error_code") if isinstance(data, dict) else None
            ok = resp.status_code == 200 and not err_code
            if ok:
                return {
                    "ok": True, "elapsed": time.monotonic() - t0,
                    "status": resp.status_code, "data": data,
                    "attempts": attempts + 1,
                }
            if err_code == 409 and attempts < max_retries:
                attempts += 1
                time.sleep(backoff + 0.05 * attempts)
                continue
            return {
                "ok": False, "elapsed": time.monotonic() - t0,
                "status": resp.status_code, "data": data,
                "attempts": attempts + 1,
            }
        except Exception as e:
            if attempts < max_retries:
                attempts += 1
                time.sleep(backoff + 0.05 * attempts)
                continue
            return {
                "ok": False, "elapsed": time.monotonic() - t0,
                "error": str(e), "status": last_status, "data": last_body,
                "attempts": attempts + 1,
            }


def _concurrent_pass(
    session: requests.Session,
    generate_url: str,
    prompts: List[str],
    is_decode: bool,
    decode_length: int,
    generate_config: Dict[str, Any],
    timeout: int,
) -> List[Dict[str, Any]]:
    bodies = [
        _build_body(p, is_decode, decode_length, generate_config) for p in prompts
    ]
    n = len(bodies)
    # one thread per request — mirrors rtp-llm's batch_perf_impl
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [
            ex.submit(_one_request, session, generate_url, b, timeout)
            for b in bodies
        ]
        results = [f.result() for f in futures]
    wall = time.monotonic() - t0
    for r in results:
        r["pass_wall"] = wall
    return results


def _summarize_prefill(
    label: str, results: List[Dict[str, Any]], seq_len: int,
) -> None:
    ok = [r for r in results if r["ok"]]
    if not ok:
        msgs = {r.get("error") or str(r.get("status")) for r in results}
        log.warning("  [%s seq=%d] FAIL errors=%s", label, seq_len, msgs)
        return
    lat = ok[0]["elapsed"]
    prefill_tps = seq_len / lat if lat > 0 else 0.0
    log.info(
        "  [%s seq=%d] TTFT=%.3fs  prefill_tps=%.0f t/s",
        label, seq_len, lat, prefill_tps,
    )


def _summarize_decode(
    label: str, results: List[Dict[str, Any]],
    batch_size: int, decode_length: int,
) -> None:
    ok = [r for r in results if r["ok"]]
    failed = len(results) - len(ok)
    if not ok:
        msgs = {r.get("error") or str(r.get("status")) for r in results}
        log.warning("  [%s bs=%d] FAIL errors=%s", label, batch_size, msgs)
        return
    wall = max(r["pass_wall"] for r in ok)
    mean_latency = sum(r["elapsed"] for r in ok) / len(ok)
    out_tokens = len(ok) * decode_length
    decode_tps = out_tokens / wall if wall > 0 else 0.0
    per_req_tps = decode_length / mean_latency if mean_latency > 0 else 0.0
    log.info(
        "  [%s bs=%d] ok=%d/%d failed=%d wall=%.3fs mean=%.3fs "
        "decode_tps=%.1f t/s per_req=%.1f t/s",
        label, batch_size, len(ok), len(results), failed, wall, mean_latency,
        decode_tps, per_req_tps,
    )


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def run_prefill_sweep(
    base_url: str,
    auth_headers: Dict[str, str],
    seq_lens: List[int],
    tokenizer,
    capture_profile: bool,
    profile_trace_prefix: str,
    profile_num_steps: int,
    generate_config: Dict[str, Any],
    timeout: int,
    skip_warmup: bool,
    profile_start_step: int = 1,
    enable_all_rank: bool = False,
) -> None:
    log.info("=" * 70)
    log.info("PREFILL sweep on %s  seq_lens=%s", base_url, seq_lens)
    log.info("=" * 70)
    generate_url = base_url.rstrip("/") + "/"

    session = _new_session(auth_headers)
    try:
        status = _fetch_worker_status(session, base_url)
        if _role(status) != "prefill":
            log.warning(
                "  seed role=%s (expected 'prefill') — continuing anyway",
                status["role"],
            )
        for seq_len in seq_lens:
            log.info("--- prefill seq_len=%d ---", seq_len)
            _pin_scheduler(session, base_url, local_batch=1, mode="prefill")

            prompt = (
                _tokenizer_prompt(tokenizer, seq_len) if tokenizer
                else _approx_prompt(seq_len)
            )
            # Prefill is 1 request per step; arm at least num_steps requests
            # so the profile window fills.
            pass_n = max(1, profile_num_steps) if capture_profile else 1
            prompts = [prompt] * pass_n

            if not skip_warmup:
                wr = _concurrent_pass(
                    session, generate_url, prompts, False, 1,
                    generate_config, timeout,
                )
                _summarize_prefill("warmup", wr, seq_len)

            mr = _concurrent_pass(
                session, generate_url, prompts, False, 1,
                generate_config, timeout,
            )
            _summarize_prefill("measure", mr, seq_len)

            if capture_profile:
                trace = "%s_prefill_seq%d" % (
                    profile_trace_prefix or "normal_profiler", seq_len,
                )
                # Prefill is one request = one engine step. If we skip the
                # first step (start_step>=1), pad the request list so that
                # at least ``start_step + num_steps`` requests arrive during
                # the window.
                pad_n = max(len(prompts), profile_start_step + profile_num_steps)
                profile_prompts = [prompts[0]] * pad_n
                _start_profile(
                    session, base_url, trace,
                    num_steps=profile_num_steps,
                    start_step=profile_start_step,
                    enable_all_rank=enable_all_rank,
                )
                pr = _concurrent_pass(
                    session, generate_url, profile_prompts, False, 1,
                    generate_config, timeout,
                )
                _summarize_prefill("profile", pr, seq_len)
                log.info(
                    "  timeline prefix=%s_wr*  (server $TORCH_CUDA_PROFILER_DIR)",
                    trace,
                )
    finally:
        session.close()


def run_decode_sweep(
    decode_urls: List[str],
    auth_headers: Dict[str, str],
    batch_sizes: List[int],
    input_len: int,
    decode_length: int,
    tokenizer,
    capture_profile: bool,
    profile_trace_prefix: str,
    profile_num_steps: int,
    generate_config: Dict[str, Any],
    timeout: int,
    skip_warmup: bool,
    traffic_url: Optional[str] = None,
    profile_start_step: int = 1,
    enable_all_rank: bool = False,
) -> None:
    """Drive decode traffic and optionally capture a torch profiler window on
    the decode workers.

    Supports multiple decode frontends (``decode_urls``) — each frontend may
    serve a subset of DP ranks on a different host. For each shape point we
    pin the scheduler and arm ``/start_profile`` on every entry so that
    * if they're in the same gRPC group, the broadcasts are redundant
      (safe no-ops);
    * if they're independent groups, each group gets armed explicitly.
      Timeline files then land on whichever host owns the ranks that ran.

    In PD-separated deployments the decode frontend doesn't accept raw ``/``
    generate (it expects a prefill→decode handoff via KV transfer). Pass
    ``traffic_url`` pointing at the prefill worker; profile arming and decode
    scheduler pin stay on ``decode_urls``.
    """
    if not decode_urls:
        log.error("run_decode_sweep called with empty decode_urls")
        return
    log.info("=" * 70)
    routing = ("via prefill %s" % traffic_url) if traffic_url else "direct"
    log.info(
        "DECODE sweep on %s  batch_sizes=%s  input_len=%d  decode_len=%d  traffic=%s",
        decode_urls, batch_sizes, input_len, decode_length, routing,
    )
    log.info("=" * 70)

    prompt = (
        _tokenizer_prompt(tokenizer, input_len) if tokenizer
        else _approx_prompt(input_len)
    )

    session = _new_session(auth_headers)
    try:
        decode_statuses: List[Dict[str, Any]] = []
        for du in decode_urls:
            st = _fetch_worker_status(session, du)
            if _role(st) != "decode":
                log.warning(
                    "  decode seed role=%s (expected 'decode') at %s — continuing",
                    st["role"], du,
                )
            decode_statuses.append(st)

        # Use the first URL's dp_size as the canonical group size; warn if
        # they disagree (would mean separate groups with different topologies).
        decode_dp = decode_statuses[0]["dp_size"]
        for du, st in zip(decode_urls[1:], decode_statuses[1:]):
            if st["dp_size"] != decode_dp:
                log.warning(
                    "  %s reports dp_size=%d but first reports %d — treating as "
                    "independent groups; bs divisibility check uses first",
                    du, st["dp_size"], decode_dp,
                )

        if traffic_url:
            prefill_status = _fetch_worker_status(session, traffic_url)
            if _role(prefill_status) != "prefill":
                log.warning(
                    "  traffic seed role=%s (expected 'prefill') — continuing",
                    prefill_status["role"],
                )
            prefill_dp = prefill_status["dp_size"]
            generate_url = traffic_url.rstrip("/") + "/"
        else:
            prefill_dp = 1
            generate_url = decode_urls[0].rstrip("/") + "/"

        for bs in batch_sizes:
            if bs % decode_dp != 0:
                log.error(
                    "  batch_size=%d not divisible by decode dp_size=%d; skip",
                    bs, decode_dp,
                )
                continue
            decode_local = bs // decode_dp
            log.info("--- decode bs=%d (decode_local=%d) ---", bs, decode_local)
            for du in decode_urls:
                _pin_scheduler(
                    session, du, local_batch=decode_local, mode="decode",
                )
            if traffic_url:
                prefill_local = max(1, bs // prefill_dp)
                _pin_scheduler(
                    session, traffic_url,
                    local_batch=prefill_local, mode="prefill",
                )

            prompts = [prompt] * bs

            if not skip_warmup:
                wr = _concurrent_pass(
                    session, generate_url, prompts, True, decode_length,
                    generate_config, timeout,
                )
                _summarize_decode("warmup", wr, bs, decode_length)

            mr = _concurrent_pass(
                session, generate_url, prompts, True, decode_length,
                generate_config, timeout,
            )
            _summarize_decode("measure", mr, bs, decode_length)

            if capture_profile:
                trace = "%s_decode_bs%d" % (
                    profile_trace_prefix or "normal_profiler", bs,
                )
                # Arm on every decode frontend (redundant if same group,
                # required if independent groups).
                for du in decode_urls:
                    _start_profile(
                        session, du, trace,
                        num_steps=profile_num_steps,
                        start_step=profile_start_step,
                        enable_all_rank=enable_all_rank,
                    )
                pr = _concurrent_pass(
                    session, generate_url, prompts, True, decode_length,
                    generate_config, timeout,
                )
                _summarize_decode("profile", pr, bs, decode_length)
                for du in decode_urls:
                    log.info(
                        "  timeline prefix=%s_wr*  on decode worker %s "
                        "($TORCH_CUDA_PROFILER_DIR)", trace, du,
                    )
    finally:
        session.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PD-separated perf + torch profiler capture driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    g = p.add_argument_group("prefill")
    g.add_argument(
        "--prefill-url",
        help="Prefill worker URL (direct, not gateway). "
             "If set, prefill sweep runs",
    )
    g.add_argument(
        "--seq-lens", default="128,256,512,1024,2048,4096,8192",
        help="Prefill seq lengths, comma-separated "
             "(default: 128,256,512,1024,2048,4096,8192)",
    )

    g = p.add_argument_group("decode")
    g.add_argument(
        "--decode-url",
        help="Decode worker URL(s) — comma-separated to cover multi-host "
             "multi-DP decode deployments. Each entry is a direct worker "
             "(not a gateway). Scheduler pin and profile arming run on every "
             "entry. If set, decode sweep runs.",
    )
    g.add_argument(
        "--batch-sizes", default="16,32,64",
        help="Decode batch sizes, comma-separated (default: 16,32,64)",
    )
    g.add_argument(
        "--input-len", type=int, default=512,
        help="Prompt length for decode sweep (default: 512)",
    )
    g.add_argument(
        "--decode-length", type=int, default=128,
        help="max_new_tokens == min_new_tokens per request (default: 128)",
    )

    g = p.add_argument_group("profile")
    g.add_argument(
        "--capture-profile", action="store_true",
        help="Arm a torch profiler window via /start_profile "
             "before each measurement traffic pass",
    )
    g.add_argument(
        "--profile-trace-name", default="normal",
        help="Trailing tag in the trace filename (default: normal). "
             "The final prefix is '<topology>_<tag>_<YYYYMMDD_HHMMSS>' "
             "where <topology> like 'P_8TP1DP-D_1TP32DP' is auto-derived "
             "from /worker_status. Use --no-topology-prefix to skip the "
             "topology part and just use '<tag>_<timestamp>'.",
    )
    g.add_argument(
        "--no-topology-prefix", action="store_true",
        help="Skip auto-probing /worker_status for tp/dp sizes; use the "
             "plain --profile-trace-name as the prefix.",
    )
    g.add_argument(
        "--profile-num-steps", type=int, default=3,
        help="Engine steps to capture per /start_profile window (default: 3). "
             "Prefill: 1 request = 1 step. Decode: 1 decode iteration = 1 step.",
    )
    g.add_argument(
        "--profile-start-step", type=int, default=1,
        help="Engine steps to skip after arming before capture starts "
             "(default: 1). /start_profile returns before the gRPC configure "
             "reaches every rank, so the first engine step can race the "
             "config and produce an empty trace — one step of buffer avoids "
             "that. Set to 0 if you know the path is warm.",
    )
    g.add_argument(
        "--enable-all-rank", action="store_true",
        help="Capture on every TP×DP rank. Default is off — only the "
             "frontend's TP rank (typically rank 0) captures, which keeps "
             "trace files small and viewable. TP ranks run symmetric kernels "
             "so rank-0 is usually enough.",
    )

    g = p.add_argument_group("common")
    g.add_argument(
        "--tokenizer-path",
        help="If set, build prompts with exact token counts via "
             "transformers tokenizer",
    )
    g.add_argument(
        "--generate-config", default="{}",
        help="Extra JSON merged into generate_config",
    )
    g.add_argument(
        "--api-key", default="",
        help="Bearer token if workers require auth",
    )
    g.add_argument("--timeout", type=int, default=600)
    g.add_argument("--skip-warmup", action="store_true")
    g.add_argument("--verbose", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.prefill_url and not args.decode_url:
        log.error("Specify --prefill-url and/or --decode-url (at least one)")
        sys.exit(1)

    try:
        gen_cfg = json.loads(args.generate_config) if args.generate_config else {}
    except json.JSONDecodeError as e:
        log.error("Invalid --generate-config JSON: %s", e)
        sys.exit(1)

    tokenizer = None
    if args.tokenizer_path:
        log.info("Loading tokenizer from %s ...", args.tokenizer_path)
        tokenizer = _load_tokenizer(args.tokenizer_path)
    else:
        log.warning(
            "No --tokenizer-path -- prompts use coarse 'hello' approximation. "
            "Pass --tokenizer-path for accurate seq_len.",
        )

    auth = {"Authorization": "Bearer %s" % args.api_key} if args.api_key else {}

    # Decode URLs are parsed up-front so the topology probe can use them
    # before the decode sweep starts. (The later decode block re-parses;
    # that's fine — this is cheap.)
    early_decode_urls: Optional[List[str]] = None
    if args.decode_url:
        early_decode_urls = [
            u.strip() for u in args.decode_url.split(",") if u.strip()
        ] or None

    # Stamp every trace name with a run id so repeated invocations produce
    # distinct timeline files on the server (rtp-llm overwrites otherwise).
    run_id = time.strftime("%Y%m%d_%H%M%S")
    tag = args.profile_trace_name
    topology = ""
    if args.capture_profile and not args.no_topology_prefix:
        topology = _probe_topology_prefix(
            auth, args.prefill_url, early_decode_urls,
        )
    if topology:
        trace_prefix = "%s_%s_%s" % (topology, tag, run_id)
    else:
        trace_prefix = "%s_%s" % (tag, run_id)
    log.info("Profile trace prefix: %s", trace_prefix)

    if args.prefill_url:
        seq_lens = _parse_int_list(args.seq_lens)
        if not seq_lens:
            log.error("--seq-lens parsed to empty list: %r", args.seq_lens)
            sys.exit(1)
        run_prefill_sweep(
            base_url=args.prefill_url, auth_headers=auth, seq_lens=seq_lens,
            tokenizer=tokenizer, capture_profile=args.capture_profile,
            profile_trace_prefix=trace_prefix,
            profile_num_steps=args.profile_num_steps,
            generate_config=gen_cfg, timeout=args.timeout,
            skip_warmup=args.skip_warmup,
            profile_start_step=args.profile_start_step,
            enable_all_rank=args.enable_all_rank,
        )
    if args.decode_url:
        batch_sizes = _parse_int_list(args.batch_sizes)
        if not batch_sizes:
            log.error("--batch-sizes parsed to empty list: %r", args.batch_sizes)
            sys.exit(1)
        decode_urls = [u.strip() for u in args.decode_url.split(",") if u.strip()]
        if not decode_urls:
            log.error("--decode-url parsed to empty list: %r", args.decode_url)
            sys.exit(1)
        # In PD-separated deployments the decode-role worker rejects raw '/'
        # generate. If --prefill-url was also provided, route decode traffic
        # through it; profile arming and decode scheduler pin stay on decode.
        run_decode_sweep(
            decode_urls=decode_urls, auth_headers=auth,
            batch_sizes=batch_sizes,
            input_len=args.input_len, decode_length=args.decode_length,
            tokenizer=tokenizer, capture_profile=args.capture_profile,
            profile_trace_prefix=trace_prefix,
            profile_num_steps=args.profile_num_steps,
            generate_config=gen_cfg, timeout=args.timeout,
            skip_warmup=args.skip_warmup,
            traffic_url=args.prefill_url if args.prefill_url else None,
            profile_start_step=args.profile_start_step,
            enable_all_rank=args.enable_all_rank,
        )


if __name__ == "__main__":
    main()
