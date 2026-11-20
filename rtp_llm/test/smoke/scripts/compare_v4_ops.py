"""Compare two CP ranks on both sides; accept only a replay of the same donor fixture."""

import json
import re
import sys
from pathlib import Path

ORDER = (
    "input",
    "attn.hc_pre.mix",
    "attn.hc_pre.post",
    "attn.hc_pre.comb",
    "attn.norm",
    "attn.rope",
    "attn.wq_a",
    "attn.qr_norm",
    "attn.wkv",
    "attn.kv_rope_gather",
    "attn.swa_write",
    "attn.swa_read",
    "attn.wq_b",
    "attn.mla.q",
    "attn.mla.kv",
    "attn.mla.indices",
    "attn.mla.length",
    "attn.mla.out",
    "attn.wo_a.fp8",
    "attn.wo_a.scale",
    "attn.wo_a",
    "attn.wo_b",
    "attn.hc_post",
    "ffn.hc_pre.mix",
    "ffn.hc_pre.post",
    "ffn.hc_pre.comb",
    "ffn.norm",
    "ffn.router",
    "ffn.pack.x",
    "ffn.pack.x_sf",
    "ffn.pack.topk_idx",
    "ffn.pack.topk_weights",
    "ffn.mega_moe",
    "ffn.combined",
    "ffn.hc_post",
    "output",
)
SHARED = tuple(
    "ffn.shared." + k for k in ("x", "x_sf", "gate_up", "hidden", "hidden_sf", "out")
)
GLOBAL = {"attn.kv_rope_gather", "attn.swa_write", "attn.swa_read", "attn.mla.kv"}
W = {
    "attn_norm",
    "ffn_norm",
    "sink",
    "q_norm",
    "kv_norm",
    "wo_a.w",
    "wo_a.sf",
    "gate.weight",
    "gate.tid2eid",
}
W.update(k + "_hc." + n for k in ("attn", "ffn") for n in ("fn", "base", "scale"))
W.update(k + "." + n for k in ("wq_a", "wq_b", "wkv", "wo_b") for n in ("w", "sf"))
W.update(
    "_" + k + "_l" + l + "_" + n
    for k in ("mega", "se")
    for l in ("1", "2")
    for n in ("w", "sf")
)


def require(test, msg):
    if not test:
        raise ValueError(msg)


def sha(v):
    return isinstance(v, str) and re.fullmatch("[0-9a-f]{64}", v)


def compare(docs):
    require(
        len(docs) == 4 and re.fullmatch("[0-9a-f]{32}", docs[0]["nonce"]), "pair nonce"
    )
    weights = []
    ab = {}
    repetitions = {}
    upstream = {}
    padding = {}
    for ix, d in enumerate(docs):
        role = ("oldfst", "newfst")[ix // 2]
        rank = ix % 2
        require(
            (d["family"], d["role"], d["rank"], d["nonce"])
            == ("v4", role, rank, docs[0]["nonce"]),
            "wrong role/rank",
        )
        require(
            sha(d["relay_sha"])
            and sha(d["natural_full_input_sha"])
            and d["consumed_full_input_sha"] == d["relay_sha"],
            "different consumed physical donor",
        )
        require(
            d["moe_executor"] in ("routed_plus_shared", "fused_se"), "invalid executor"
        )
        ops = d["ops"]
        require(
            set(ops)
            == set(ORDER)
            | (set(SHARED) if d["moe_executor"] == "routed_plus_shared" else set())
            and set(d["weights"]) == W,
            "incomplete operators or weights",
        )
        require(
            all(
                sha(w["sha"]) and len(w["shape"]) == len(w["stride"])
                for w in d["weights"].values()
            ),
            "weight metadata",
        )
        loc = ops["input"]["pos"]
        require(
            len(set(loc)) == len(loc)
            and all(type(p) == int and 0 <= p < 55 for p in loc),
            "rank positions",
        )
        natural, donor = d["natural_valid_rows"], ops["input"]
        require(
            all(natural[k] == donor[k] for k in ("pos", "dtype", "shape"))
            and len(natural["rows"]) == len(loc)
            and all(map(sha, natural["rows"]))
            and d["valid_input_rows"] == len(loc)
            and len(loc) <= d["physical_input_rows"] <= 28,
            "natural input coverage",
        )
        key = role + str(rank)
        upstream[key] = [
            p for p, l, r in zip(loc, natural["rows"], donor["rows"]) if l != r
        ]
        require(
            d["natural_rows_equal_relay"] is (not upstream[key]),
            "natural witness inconsistent",
        )
        padding[key] = {
            "physical": d["physical_input_rows"],
            "valid": len(loc),
            "full_natural_equals_donor": d["natural_full_input_sha"] == d["relay_sha"],
        }
        for k, v in ops.items():
            require(
                v["pos"] == (list(range(55)) if k in GLOBAL else loc)
                and len(v["pos"]) == len(v["rows"])
                and all(map(sha, v["rows"])),
                "row coverage",
            )
        require(
            all(
                ops[k]["rows"] == ops["attn.swa_read"]["rows"]
                for k in ("attn.kv_rope_gather", "attn.swa_write", "attn.mla.kv")
            ),
            "fresh SWA read not proven",
        )
        require(set(d["ab"]) == {"wq_a", "wkv"}, "missing isolated Q/KV replay")
        for k, v in d["ab"].items():
            require(
                all(sha(v[x]) for x in ("live", "independent", "shared"))
                and all(
                    type(v[x]) is bool for x in ("activation_equal", "scales_equal")
                ),
                "Q/KV replay",
            )
        ab[role + str(rank)] = {
            k: {
                "activation_equal": v["activation_equal"],
                "scales_equal": v["scales_equal"],
                "independent_equals_shared": v["independent"] == v["shared"],
                "live_equals_independent": v["live"] == v["independent"],
                "live_equals_shared": v["live"] == v["shared"],
            }
            for k, v in d["ab"].items()
        }
        repeat_keys = [
            k for k in ops if k.startswith("ffn.") and not k.startswith("ffn.hc")
        ]
        require(
            set(d["repeat_moe"]) == set(repeat_keys), "incomplete same-input repeat"
        )
        repetitions[role + str(rank)] = {}
        for k in repeat_keys:
            a, b = ops[k], d["repeat_moe"][k]
            require(
                all(a[x] == b[x] for x in ("pos", "shape", "dtype"))
                and len(b["rows"]) == len(a["rows"])
                and all(map(sha, b["rows"])),
                "repeat layout",
            )
            points = [p for p, l, r in zip(a["pos"], a["rows"], b["rows"]) if l != r]
            if points:
                repetitions[role + str(rank)][k] = points
    for ix in (0, 2):
        p = docs[ix]["ops"]["input"]["pos"] + docs[ix + 1]["ops"]["input"]["pos"]
        require(sorted(p) == list(range(55)), "missing physical CP rank")
    for rank in (0, 1):
        a, b = docs[rank], docs[rank + 2]
        require(
            a["relay_sha"] == b["relay_sha"] and a["ops"]["input"] == b["ops"]["input"],
            "different donor inputs",
        )
        weights += [
            {"rank": rank, "name": k}
            for k in sorted(W)
            if a["weights"][k] != b["weights"][k]
        ]
    first = None
    first40 = None
    same_path = len({d["moe_executor"] for d in docs}) == 1
    extra = (
        SHARED if same_path and docs[0]["moe_executor"] == "routed_plus_shared" else ()
    )
    for k in (
        ORDER[: ORDER.index("ffn.router")] + extra + ORDER[ORDER.index("ffn.router") :]
    ):
        pos = set()
        for rank in (0, 1):
            a, b = (d["ops"][k] for d in (docs[rank], docs[rank + 2]))
            require(
                (a["dtype"], a["shape"], a["pos"])
                == (b["dtype"], b["shape"], b["pos"]),
                "incompatible operand layout",
            )
            pos.update(p for p, x, y in zip(a["pos"], a["rows"], b["rows"]) if x != y)
        if first is None and pos:
            first = {"stage": k, "positions": sorted(pos)}
        if first40 is None and 40 in pos:
            first40 = k
    return {
        "valid": True,
        "same_donor_replayed_on_both": True,
        "fresh_kv_only": True,
        "runtime_weight_differences": weights,
        "natural_input_differing_positions_vs_donor": upstream,
        "physical_vs_valid_input": padding,
        "same_effective_weight_bytes_and_layout": not weights,
        "first_different_observed_boundary": first,
        "first_different_boundary_at_position40": first40,
        "same_moe_executor": same_path,
        "qkv_isolated_replay": ab,
        "same_input_moe_repeat_differences": repetitions,
        "interpretation": "Fused kernels are single boundaries; differing weights/layout or unstable same-input repeats preclude deterministic loader attribution.",
    }


if __name__ == "__main__":
    try:
        require(len(sys.argv) == 5, "need old0 old1 new0 new1")
        result = compare([json.loads(Path(p).read_bytes()) for p in sys.argv[1:]])
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        print(
            json.dumps(
                {
                    "valid": False,
                    "reason": "incomplete or unmatched V4 operator evidence",
                }
            )
        )
        sys.exit(2)
    print(json.dumps(result, separators=(",", ":")))
