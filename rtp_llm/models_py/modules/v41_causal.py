"""Paired ARM test only: V4.1 layer 1, known 783-token request with exactly 512 reused."""

import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

TARGET = "//rtp_llm/test/smoke:deepseek_v41_cache_rounds_sm100_arm"
KEYS = {"positions", "incoming", "previous_mix", "swa_tail"}


def _digest(x):
    x = x.detach()
    sha = hashlib.sha256()
    # A routed expert pack can be multi-GiB. Never materialize a second pack.
    stride = max(1, x[0].numel() * x.element_size()) if x.ndim and len(x) else 1
    for part in (
        x.reshape(1).split(1) if x.ndim == 0 else x.split(max(1, (8 << 20) // stride))
    ):
        sha.update(part.contiguous().view(torch.uint8).cpu().numpy())
    return {
        "dtype": str(x.dtype),
        "shape": list(x.shape),
        "sha256": sha.hexdigest(),
    }


def _schema(data, meta, nonce, rank):
    if meta != dict(
        nonce=nonce, role="oldfst", family="v41", rank=str(rank), stage="layer1.hit512"
    ):
        raise ValueError("wrong V4.1 fixture provenance")
    if set(data) != KEYS:
        raise ValueError("wrong V4.1 fixture fields")
    pos = data["positions"]
    if pos.ndim != 1 or pos.dtype != torch.int64 or not 1 <= len(pos) <= 271:
        raise ValueError("invalid V4.1 fixture CP extent")
    ids = pos.tolist()
    if len(set(ids)) != len(ids) or not all(512 <= p < 783 for p in ids):
        raise ValueError("invalid V4.1 fixture logical positions")
    for key, shape, dtype in (
        ("incoming", (len(ids), 4, 5120), torch.bfloat16),
        ("previous_mix", (len(ids), 4), torch.float32),
        ("swa_tail", (127, 512), torch.bfloat16),
    ):
        if tuple(data[key].shape) != shape or data[key].dtype != dtype:
            raise ValueError("invalid V4.1 fixture tensor " + key)


def _read(rank, nonce, positions):
    from safetensors import safe_open

    path = (
        Path(os.environ["TEST_SRCDIR"])
        / os.environ["TEST_WORKSPACE"]
        / (f"rtp_llm/test/smoke/arm_causal_inputs/v41_r{rank}.safetensors")
    )
    if not path.is_file() or not 0 < path.stat().st_size < 12 << 20:
        raise ValueError("missing or oversized V4.1 private fixture")
    with safe_open(str(path), framework="pt", device="cpu") as f:
        if set(f.keys()) != KEYS:
            raise ValueError("unexpected V4.1 private fixture fields")
        data = {k: f.get_tensor(k) for k in KEYS}
        _schema(data, f.metadata(), nonce, rank)
    if not torch.equal(data["positions"], positions):
        raise ValueError("paired V4.1 CP rank owns different rows")
    return data


def observe(trace, call, layer, cp, h, *args, **kwargs):
    role, nonce = os.environ["RTP_V41_CAUSAL_ROLE"], os.environ["RTP_ARM_PAIR_NONCE"]
    rank = int(cp.cp_rank)
    if (
        role not in ("oldfst", "newfst")
        or not re.fullmatch("[0-9a-f]{32}", nonce)
        or os.getenv("RTP_V41_EXPECTED_LOADER") != role
        or os.getenv("TEST_TARGET") != TARGET
        or trace.phase != "hit"
        or trace.prefix != 512
        or int(cp.cp_size) != 2
        or rank not in (0, 1)
        or layer.attn.layer_id != 1
        or layer.attn.compress_ratio != 0
        or h.dtype != torch.bfloat16
        or h.shape[1:] != (4, 5120)
    ):
        raise ValueError("unexpected V4.1 causal fixture scope")
    valid = cp.local_is_real.reshape(-1).bool()
    positions = cp.global_positions.reshape(-1)[valid].long().cpu()
    marker = trace.root / "v41_causal_replay_id"
    replay = marker.is_file()
    if replay and marker.read_text() != trace.run:
        raise ValueError("mismatched V4.1 repeat request")
    mode = "replay" if replay else "natural"
    incoming = _read(rank, nonce, positions) if replay and role == "newfst" else {}
    relay = {"positions": positions}
    record = dict(
        schema=1,
        family="v41",
        nonce=nonce,
        role=role,
        mode=mode,
        rank=rank,
        positions=positions.tolist(),
        events=[],
        weights={},
        skipped=[],
    )

    def event(name, value, *, layout="local"):
        # Do not hash physical padding / invalid Q rows. All exported observations are digests.
        if layout == "local":
            if value.size(0) == len(valid):
                value = value[valid]
            elif value.size(0) != len(positions):
                raise ValueError("unmappable V4.1 rank rows: " + name)
        summary = _digest(value)
        if layout == "local":
            rows = (
                value.detach()
                .contiguous()
                .view(torch.uint8)
                .reshape(len(positions), -1)
                .cpu()
                .numpy()
            )
            summary["rows"] = [hashlib.sha256(row).hexdigest() for row in rows]
        record["events"].append([name, summary])

    def boundary(name, value):
        event(name + ".observed", value)
        if not replay:
            return value
        # Only valid request rows are serialized. Canonicalize padding on BOTH
        # replay arms, without mutating the ordinary request or the source tensor.
        out = torch.zeros_like(value)
        live = value[valid].detach().contiguous()
        if role == "oldfst":
            relay[name] = live.cpu()
        else:
            live = incoming[name].to(value.device)
        out[valid] = live
        event(name + ".used", out)
        return out

    def weight(name, value):
        if isinstance(value, torch.Tensor):
            record["weights"][name] = _digest(value)
        else:
            record["skipped"].append("weight:" + name)

    # Fingerprint actual device-bound native tensors, not checkpoint aliases.
    for key, owner in (
        ("engram_wkv", layer.engram.wkv),
        ("wq_b", layer.attn.wq_b),
        ("wo_b", layer.attn.wo_b),
    ):
        native = getattr(owner, "v41_runtime_linear", owner)
        for field in ("weight", "weight_scales"):
            weight(key + "." + field, getattr(native, field, None))
    fused = getattr(layer.attn, "wq_a_wkv", None)
    if fused is not None:
        native = getattr(fused, "v41_runtime_linear", fused)
        for field in ("weight", "weight_scales"):
            weight("fused_qkv." + field, getattr(native, field, None))
    for label, obj, names in (
        ("engram", layer.engram, ("q_weight", "k_weight")),
        ("attn_hc", layer.attn_hc, ("fn", "base", "scale")),
        ("ffn_hc", layer.ffn_hc, ("fn", "base", "scale")),
        ("attn_norm", layer.attn_norm, ("weight",)),
        ("ffn_norm", layer.ffn_norm, ("weight",)),
        (
            "attention",
            layer.attn,
            ("q_norm", "kv_norm", "attn_sink", "_wo_a_stk_w", "_wo_a_stk_s"),
        ),
    ):
        for field in names:
            weight(label + "." + field, getattr(obj, field, None))
    gate = layer.ffn.gate
    weight("gate.bf16", gate._weight_bf16())
    for field in ("bias", "bias_vl", "tid2eid"):
        value = getattr(gate, field, None)
        if value is not None:
            weight("gate." + field, value)
    table = layer.engram.embed_tokens
    # Only rows named by the matched request (never read or relay the whole table).
    ids = layer.engram_hashes[valid].detach().long().cpu().reshape(-1)
    ids = torch.unique(ids[ids >= 0], sorted=True)
    if (
        len(ids) > 271 * 24
        or table.weight.element_size() != 1
        or table.scales.element_size() != 1
    ):
        raise ValueError("unexpected Engram fixture lookup")
    weight("engram.addresses", ids)
    for field in ("weight", "scales"):
        weight(
            "engram.selected." + field,
            getattr(table, field).view(torch.uint8).index_select(0, ids),
        )
    experts = layer.ffn.fused_moe.fused_experts
    for field in ("_mega_l1_w", "_mega_l1_sf", "_mega_l2_w", "_mega_l2_sf"):
        weight("experts." + field, getattr(experts, field, None))
    if layer.ffn.fused_moe.includes_shared_expert:
        for field in ("_se_l1_w", "_se_l1_sf", "_se_l2_w", "_se_l2_sf"):
            weight("experts." + field, getattr(experts, field, None))
    else:
        for key in ("w13", "w2"):
            linear = getattr(layer.ffn.shared_experts, key, None)
            for field in ("weight", "weight_scales"):
                weight("shared." + key + "." + field, getattr(linear, field, None))

    event("input_ids", args[0])
    event("hash_ids", layer.engram_hashes)
    event("token_mask", layer.engram_token_mask)
    previous = layer.attn_hc._previous_ref()
    h = boundary("incoming", h)
    mix = boundary("previous_mix", previous.pre_mix_out)
    with ExitStack() as stack:
        if replay:
            stack.enter_context(patch.object(previous, "pre_mix_out", mix))

        def wrap(obj, attr, intercept):
            fn = getattr(obj, attr)

            def wrapped(*a, **k):
                return intercept(fn, a, k)

            stack.enter_context(patch.object(obj, attr, wrapped))

        for key, module in (
            ("engram_wkv", layer.engram.wkv),
            ("engram", layer.engram),
            ("attn_norm", layer.attn_norm),
            ("attention", layer.attn),
            ("ffn_norm", layer.ffn_norm),
            ("moe", layer.ffn),
        ):
            stack.callback(
                module.register_forward_pre_hook(
                    lambda _m, a, n=key: event(n + ".in", a[0])
                ).remove
            )
            stack.callback(
                module.register_forward_hook(
                    lambda _m, _a, out, n=key: event(n + ".out", out)
                ).remove
            )

        def hc(before):
            def run(fn, a, k):
                y, post, comb = fn(*a, **k)
                for label, value in (("readout", y), ("post", post), ("comb", comb)):
                    event(before + "." + label, value)
                return y, post, comb

            return run

        wrap(layer.attn_hc, "pre", hc("attn_hc"))
        wrap(layer.ffn_hc, "pre", hc("ffn_hc"))

        def qkv(fn, a, k):
            common = a[1]
            event("rope", common.freqs_cis)
            value = fn(*a, **k)
            event("qr", value.qr)
            event("new_kv", value.kv_full, layout="global")
            return value

        wrap(layer.attn, "_prefill_compute_qkv", qkv)

        def pre_qkv(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                event("fused_qkv.qr", result[0].squeeze(0))
                event("fused_qkv.kv", result[1].squeeze(0))
            else:
                for key in ("wq_a", "wkv"):
                    op = getattr(layer.attn, key)
                    op = getattr(op, "v41_runtime_linear", op)
                    for field in ("weight", "weight_scales"):
                        weight(key + "." + field, getattr(op, field, None))
            return result

        wrap(layer.attn, "_try_fused_qr_kv", pre_qkv)

        def packed(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                event("gate.scores", result.scores)
            return result

        wrap(gate, "prepare_gate_payload", packed)

        def expert_pack(fn, a, k):
            result = fn(*a, **k)
            y, weights, ids = result
            event("gate.packed_weights", weights)
            event("gate.packed_ids", ids)
            event("routed.out", y)
            return result

        wrap(experts, "forward_gate_pack", expert_pack)

        def swa(fn, a, k):
            common = a[1]
            meta = common.swa_meta
            if (
                common.batch_size != 1
                or not common.cp_on
                or not common.any_cont
                or layer.attn._host_prefill_prefixes(common) != [512]
                or layer.attn._host_prefill_lengths(common) != [271]
                or meta.cache_gather_lens.reshape(-1).tolist() != [127]
            ):
                raise ValueError("unexpected V4.1 live SWA prefix geometry")
            out = fn(*a, **k)
            prefix = out[0, :127]
            if prefix.shape != (127, 512) or prefix.dtype != torch.bfloat16:
                raise ValueError("invalid V4.1 decoded SWA operand")
            event("swa_tail.observed", prefix, layout="global")
            if replay:
                if role == "oldfst":
                    relay["swa_tail"] = prefix.detach().cpu().contiguous()
                else:
                    prefix.copy_(incoming["swa_tail"])
            event("swa_tail.used", prefix, layout="global")
            return out

        wrap(layer.attn, "_swa_prefill_concat", swa)

        def query(fn, a, k):
            value = fn(*a, **k)
            event("q_rope", value)
            return value

        wrap(layer.attn, "_project_prefill_q", query)

        def flash_inputs(fn, a, k):
            event("flash.indices", k["indices"])
            event("flash.lens", k["topk_length"])
            return fn(*a, **k)

        wrap(layer.attn, "_prefill_sparse_attention", flash_inputs)

        def proj(fn, a, k):
            event("flash.out", a[0])
            result = fn(*a, **k)
            event("o_proj", k["out"])
            return result

        wrap(layer.attn, "_prefill_output_proj_into", proj)

        def mega(fn, a, k):
            result = fn(*a, **k)
            if result is not None:
                for label, t in zip(("residual", "norm", "post", "comb"), result):
                    event("mega_mhc." + label, t)
            return result

        wrap(layer, "_try_mega_mhc", mega)
        out = call(h, *args, **kwargs)
    event("outgoing", out)
    event("next_mix", layer.ffn_hc.pre_mix_out)
    with (trace.root / f"v41_causal_{mode}_r{rank}.json").open("x") as f:
        json.dump(record, f, separators=(",", ":"))
    if replay and role == "oldfst":
        from safetensors.torch import save_file

        provenance = dict(
            nonce=nonce, role=role, family="v41", rank=str(rank), stage="layer1.hit512"
        )
        _schema(relay, provenance, nonce, rank)
        private = trace.root / "arm_causal_private"
        private.mkdir(mode=0o700, exist_ok=True)
        output = private / f"v41_r{rank}.safetensors"
        if output.exists():
            raise ValueError("duplicate V4.1 private fixture")
        save_file(relay, str(output), metadata=provenance)
    return out
