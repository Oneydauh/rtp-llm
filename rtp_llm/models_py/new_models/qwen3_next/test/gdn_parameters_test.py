"""Exercise the parameter owner with on-disk fused and split HF checkpoints."""
import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from rtp_llm.models_py.new_models.qwen3_next.gdn_parameters import Qwen3NextGdnParameters


class GdnOwnershipTest(unittest.TestCase):
    def model(self, rank):
        config = SimpleNamespace(linear_key_head_dim=2, linear_value_head_dim=2,
            linear_num_key_heads=2, linear_num_value_heads=4, linear_conv_kernel_dim=3)
        return Qwen3NextGdnParameters(config, 8, 1e-6, 2, rank, torch.float32, None, "linear_attn")

    def test_per_head_and_split_produce_same_projections(self):
        matrix = torch.arange(24 * 8, dtype=torch.float32).reshape(24, 8)
        q, k, v, z = matrix.split([4, 4, 8, 8])
        ba = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        b, a = ba.chunk(2)
        fused = {
            "in_proj_qkvz.weight": torch.cat([t.reshape(2, -1, 8) for t in (q, k, v, z)], dim=1).reshape(-1, 8),
            "in_proj_ba.weight": torch.cat([t.reshape(2, -1, 8) for t in (b, a)], dim=1).reshape(-1, 8),
        }
        split = {"in_proj_qkv.weight": torch.cat([q, k, v]), "in_proj_z.weight": z,
                 "in_proj_b.weight": b, "in_proj_a.weight": a}
        conv = torch.arange(16 * 3, dtype=torch.float32).reshape(16, 1, 3)
        out = torch.arange(64, dtype=torch.float32).reshape(8, 8)
        common = {"A_log": torch.arange(4, dtype=torch.float32),
                  "dt_bias": torch.arange(4, dtype=torch.float32) + 10,
                  "conv1d.weight": conv, "norm.weight": torch.ones(2), "out_proj.weight": out}
        for layout in (fused, split):
            with tempfile.TemporaryDirectory() as folder:
                path = os.path.join(folder, "model.safetensors")
                save_file({**layout, **common}, path)
                for rank in range(2):
                    owner = self.model(rank)
                    with safe_open(path, framework="pt", device="cpu") as weights:
                        for name in reversed(weights.keys()):
                            owner.load_weights({name: weights.get_tensor(name)})
                    owner.validate_weights_loaded()
                    local = lambda values: torch.cat([part.chunk(2)[rank] for part in values])
                    torch.testing.assert_close(owner.in_proj_qkvz.weight, local((q, k, v, z)))
                    torch.testing.assert_close(owner.in_proj_ba_w, local((b, a)).T)
                    torch.testing.assert_close(owner.conv1d_w, local(conv.squeeze(1).split((4, 4, 8))))
                    torch.testing.assert_close(owner.dt_bias, common["dt_bias"].chunk(2)[rank])
                    torch.testing.assert_close(owner.a_log, common["A_log"].chunk(2)[rank])
                    torch.testing.assert_close(owner.out_proj.weight, out.chunk(2, dim=1)[rank])

    def test_mixed_layout_fails(self):
        m = self.model(0)
        m.load_weights({"in_proj_b.weight": torch.ones(4, 8)})
        with self.assertRaisesRegex(RuntimeError, "mix"):
            m.load_weights({"in_proj_ba.weight": torch.ones(8, 8)})


if __name__ == "__main__":
    unittest.main()
