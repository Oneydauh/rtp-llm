"""CPU-only protocol and live-call wiring; not a substitute for paired GB200 inference."""

import json
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from rtp_llm.models_py.modules import v41_causal as trace
from rtp_llm.test.smoke.scripts.compare_v41_causal import CORE, compare


class Identity(torch.nn.Module):
    def forward(self, x, *args):
        return x


class HC(Identity):
    def pre(self, x):
        return x[:, 0], x[..., :1], x[..., :4]


class Attn(Identity):
    layer_id = 1
    compress_ratio = 0

    def _swa_prefill_concat(self, qkv, common):
        return torch.full((1, 398, 512), self.seed, dtype=torch.bfloat16)

    def _prefill_compute_qkv(self, x, common):
        return NS(qr=x, kv_full=x)

    def _project_prefill_q(self, x):
        return x

    def _prefill_sparse_attention(self, q, *, indices, topk_length):
        return q

    def _prefill_output_proj_into(self, x, *, out):
        out.copy_(x)

    def _try_fused_qr_kv(self, *args):
        return None

    def _host_prefill_prefixes(self, _):
        return [512]

    def _host_prefill_lengths(self, _):
        return [271]


class CausalTest(unittest.TestCase):
    def test_real_call_private_transport(self):
        from safetensors import safe_open

        with TemporaryDirectory() as a, TemporaryDirectory() as b, TemporaryDirectory() as tmp:
            nonce, run = "9" * 32, "a" * 32
            fixture = Path(tmp) / "w/rtp_llm/test/smoke/arm_causal_inputs"
            fixture.mkdir(parents=True)

            def invoke(role, root, seed):
                root = Path(root)
                rank = 0
                ids = torch.tensor([2, 4])
                valid = torch.tensor([True, False])
                cp = NS(
                    cp_size=2,
                    cp_rank=rank,
                    local_is_real=valid,
                    global_positions=torch.tensor([639, 1]),
                )
                h = torch.full((2, 4, 5120), seed, dtype=torch.bfloat16)
                prev = NS(pre_mix_out=torch.full((2, 4), seed, dtype=torch.float32))
                hc = HC()
                hc._previous_ref = lambda: prev
                net = NS(
                    attn=Attn(),
                    attn_hc=hc,
                    ffn_hc=HC(),
                    ffn_norm=Identity(),
                    attn_norm=Identity(),
                    engram=Identity(),
                    ffn=Identity(),
                    engram_token_mask=valid,
                    engram_hashes=torch.zeros((2, 24), dtype=torch.long),
                )
                net.engram.wkv = Identity()
                net.engram.embed_tokens = NS(
                    weight=torch.zeros((1, 256), dtype=torch.uint8),
                    scales=torch.zeros((1, 8), dtype=torch.uint8),
                )
                net.attn.seed = seed
                for field in ("wq_a", "wkv", "wq_b", "wo_b"):
                    setattr(net.attn, field, Identity())
                net.ffn.gate = NS(
                    _weight_bf16=lambda: h[0], prepare_gate_payload=lambda: None
                )
                net.ffn.fused_moe = NS(
                    fused_experts=NS(forward_gate_pack=lambda: None),
                    includes_shared_expert=False,
                )
                net.ffn.shared_experts = None
                net._try_mega_mhc = lambda *args: None
                net.ffn_hc.pre_mix_out = prev.pre_mix_out
                state = NS(root=root, run=run, phase="hit", prefix=512)
                (root / "v41_causal_replay_id").write_text(run)
                used = {}

                def block(x, input_ids):
                    common = NS(
                        batch_size=1,
                        cp_on=True,
                        any_cont=True,
                        swa_meta=NS(cache_gather_lens=torch.tensor([127])),
                    )
                    buf = net.attn._swa_prefill_concat(None, common)
                    used.update(
                        x=x.clone(), mix=prev.pre_mix_out.clone(), raw=buf.clone()
                    )
                    return x

                with patch.dict(
                    os.environ,
                    dict(
                        RTP_V41_CAUSAL_ROLE=role,
                        RTP_ARM_PAIR_NONCE=nonce,
                        RTP_V41_EXPECTED_LOADER=role,
                        TEST_TARGET=trace.TARGET,
                        TEST_SRCDIR=tmp,
                        TEST_WORKSPACE="w",
                    ),
                ):
                    trace.observe(state, block, net, cp, h, ids)
                self.assertEqual(prev.pre_mix_out[0, 0].item(), seed)
                return used

            old = invoke("oldfst", a, 3)
            old_path = Path(a) / "arm_causal_private/v41_r0.safetensors"
            with safe_open(str(old_path), framework="pt") as f:
                self.assertEqual(set(f.keys()), trace.KEYS)
                self.assertEqual(f.get_tensor("incoming").shape, (1, 4, 5120))
            (fixture / "v41_r0.safetensors").symlink_to(old_path)
            new = invoke("newfst", b, 9)
            self.assertFalse((Path(b) / "arm_causal_private").exists())
            self.assertTrue(
                torch.equal(old["x"], new["x"]) and torch.equal(old["mix"], new["mix"])
            )
            self.assertTrue(torch.equal(old["raw"][:, :127], new["raw"][:, :127]))
            self.assertEqual(new["raw"][0, 127, 0].item(), 9)  # fresh KV never replaced
            self.assertEqual(new["x"][1].count_nonzero().item(), 0)
            with patch.dict(os.environ, dict(TEST_SRCDIR=tmp, TEST_WORKSPACE="w")):
                with self.assertRaisesRegex(ValueError, "provenance"):
                    trace._read(0, "0" * 32, torch.tensor([639]))
                with self.assertRaisesRegex(ValueError, "different rows"):
                    trace._read(0, nonce, torch.tensor([640]))

    def test_compare_rejects_nonidentical_replay(self):
        with TemporaryDirectory() as tmp:
            roots = [Path(tmp) / k for k in ["oldfst", "newfst"]]
            for root in roots:
                root.mkdir()
                for rank in range(2):
                    pos = list(range(512 + rank, 783, 2))
                    for mode in ("natural", "replay"):
                        val = dict(
                            sha256="2" * 64, shape=[len(pos), 4], dtype="torch.float32"
                        )
                        names = set(CORE)
                        if mode == "replay":
                            names.update(("incoming.used", "previous_mix.used"))
                        rec = dict(
                            schema=1,
                            family="v41",
                            nonce="7" * 32,
                            mode=mode,
                            rank=rank,
                            role=root.name,
                            positions=pos,
                            weights={"w": val},
                            skipped=[],
                            events=[[s, val] for s in sorted(names)],
                        )
                        (root / f"v41_causal_{mode}_r{rank}.json").write_text(
                            json.dumps(rec)
                        )
            self.assertEqual(compare(*roots)["status"], "compared")
            target = roots[1] / "v41_causal_replay_r1.json"
            doc = json.loads(target.read_text())
            dict(doc["events"])["incoming.used"]["sha256"] = "d" * 64
            target.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "not shared"):
                compare(*roots)


if __name__ == "__main__":
    unittest.main()
