"""Compare public per-rank causal reports. Never opens the private relay."""

import argparse
import json
import re
from pathlib import Path

CORE = {
    "input_ids",
    "hash_ids",
    "token_mask",
    "incoming.observed",
    "previous_mix.observed",
    "engram.in",
    "engram_wkv.in",
    "engram_wkv.out",
    "engram.out",
    "attn_hc.readout",
    "attn_hc.post",
    "attn_hc.comb",
    "attn_norm.in",
    "attn_norm.out",
    "attention.in",
    "rope",
    "qr",
    "new_kv",
    "swa_tail.observed",
    "swa_tail.used",
    "q_rope",
    "flash.indices",
    "flash.lens",
    "flash.out",
    "o_proj",
    "attention.out",
    "moe.in",
    "moe.out",
    "outgoing",
    "next_mix",
}
FIXED = ("incoming.used", "previous_mix.used", "swa_tail.used")


def read(root, role, mode, rank):
    d = json.loads((root / f"v41_causal_{mode}_r{rank}.json").read_text())
    if (
        d.get("schema"),
        d.get("family"),
        d.get("role"),
        d.get("rank"),
        d.get("mode"),
    ) != (1, "v41", role, rank, mode) or not re.fullmatch("[a-f0-9]{32}", d["nonce"]):
        raise ValueError("invalid V4.1 causal manifest")
    pos = d["positions"]
    if (
        not pos
        or len(set(pos)) != len(pos)
        or not all(type(n) is int and 512 <= n < 783 for n in pos)
    ):
        raise ValueError("invalid V4.1 positions")
    seq = [key for key, value in d["events"]]
    required = CORE.union(FIXED) if mode == "replay" else CORE
    if not required.issubset(seq) or len(set(seq)) != len(seq):
        raise ValueError(
            "missing or ambiguous stage: " + ",".join(sorted(required.difference(seq)))
        )
    if not d["weights"] or d["skipped"]:
        raise ValueError("runtime weights incomplete: " + ",".join(d["skipped"]))
    checks = [v for _, v in d["events"]] + list(d["weights"].values())
    for v in checks:
        if (
            not re.fullmatch("[0-9a-f]{64}", v["sha256"])
            or not v["dtype"]
            or not isinstance(v["shape"], list)
        ):
            raise ValueError("invalid V4.1 fingerprint")
        if "rows" in v and (
            len(v["rows"]) != len(pos)
            or not all(re.fullmatch("[0-9a-f]{64}", i) for i in v["rows"])
        ):
            raise ValueError("invalid V4.1 row fingerprints")
    return d


def compare(old, new):
    reports = [{"rank": r} for r in range(2)]
    nonces = set()
    for mode in ("natural", "replay"):
        positions = []
        for rank in (0, 1):
            a, b = (
                read(p, role, mode, rank)
                for p, role in ((old, "oldfst"), (new, "newfst"))
            )
            nonces.update((a["nonce"], b["nonce"]))
            if (
                a["positions"] != b["positions"]
                or a["weights"].keys() != b["weights"].keys()
            ):
                raise ValueError("different CP or weight layout")
            positions.extend(a["positions"])
            weights = [k for k in a["weights"] if a["weights"][k] != b["weights"][k]]
            left, right = dict(a["events"]), dict(b["events"])
            if mode == "replay" and any(left[k] != right[k] for k in FIXED):
                raise ValueError(
                    "Old layer inputs, delayed mix or decoded prefix not shared in replay"
                )
            first = None
            skip = (
                {"incoming.observed", "previous_mix.observed", "swa_tail.observed"}
                if mode == "replay"
                else set()
            )
            for (ka, va), (kb, vb) in zip(a["events"], b["events"]):
                if ka != kb:
                    first = {"execution_path_old": ka, "execution_path_new": kb}
                    break
                if va != vb and ka not in skip:
                    rows = []
                    if "rows" in va and "rows" in vb:
                        rows = [
                            p
                            for p, x, y in zip(a["positions"], va["rows"], vb["rows"])
                            if x != y
                        ]
                    first = {
                        "stage": ka,
                        "first_logical_position": min(rows) if rows else None,
                    }
                    break
            if first is None and len(left) != len(right):
                raise ValueError("unmatched trailing V4.1 stages")
            reports[rank][mode] = {
                "first_observed_difference": first,
                "differing_native_weights": weights,
                "causal_interpretation": (
                    "weight_or_address_difference_do_not_infer_kernel"
                    if weights
                    else (
                        "different_execution_paths"
                        if first and "execution_path_old" in first
                        else (
                            "first_observed_boundary_not_kernel_root"
                            if first
                            else "no_difference_at_recorded_boundaries"
                        )
                    )
                ),
            }
        if sorted(positions) != list(range(512, 783)):
            raise ValueError("incomplete CP coverage")
    if len(nonces) != 1:
        raise ValueError("reports do not belong to the same paired invocation")
    return dict(
        status="compared",
        family="v41",
        nonce=next(iter(nonces)),
        ranks=reports,
        interpretation="Observed boundaries; absence of a difference is not proof for other layers.",
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("old", type=Path)
    cli.add_argument("new", type=Path)
    args = cli.parse_args()
    try:
        result = compare(args.old, args.new)
    except (OSError, KeyError, ValueError, TypeError) as error:
        print(json.dumps(dict(status="incomplete", family="v41", reason=str(error))))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
