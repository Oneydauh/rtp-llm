"""Actual layer-local Engram checkpoint loads (no GPU/UVA emulation)."""

import gc
import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.modules.dsv41.engram import Engram
from rtp_llm.models_py.new_models.deepseek_v41.engram import OwnedEngram


class EngramSourceTest(unittest.TestCase):
    cfg = dict(
        engram_layer_ids=[1, 14],
        engram_num_embeddings=[8, 12],
        engram_max_ngram_size=2,
        engram_n_heads=1,
        engram_head_dim=32,
        engram_compressed_vocab_size=8,
        engram_pad_token_id=0,
        engram_vocab_size=7,
        hidden_size=32,
        hc_mult=2,
        rms_norm_eps=1e-6,
    )

    def fixture(self, directory, *, bad=None):
        index = {}
        for layer_id, rows in ((1, 8), (14, 12)):
            prefix = f"layers.{layer_id}.engram."
            name = f"layer-{layer_id}.safetensors"
            tensors = {
                "embed.weight": (
                    torch.arange(rows * 32).reshape(rows, 32) % 7 + layer_id
                ).to(torch.float8_e4m3fn),
                "embed.scale": torch.full((rows, 1), 126, dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
                "wkv.weight": ((torch.arange(96 * 32).reshape(96, 32) % 9) - 4).to(
                    torch.float8_e4m3fn
                ),
                "wkv.scale": torch.full((3, 1), 125, dtype=torch.uint8).view(
                    torch.float8_e8m0fnu
                ),
                "q_weight": torch.full((2, 32), 0.5, dtype=torch.bfloat16),
                "k_weight": torch.full((2, 32), 1.5, dtype=torch.bfloat16),
            }
            if bad is not None:
                bad(tensors)
            save_file({prefix + k: v for k, v in tensors.items()}, directory / name)
            index.update({prefix + k: name for k in tensors})
        (directory / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": index})
        )

    def load(self, folder, layer, method):
        return OwnedEngram.from_checkpoint(
            self.cfg, layer, folder, "cpu", load_method=method
        )

    def test_native_owned_values_equal_main_after_close_and_allocator_churn(self):
        import fastsafetensors.cpp as native

        self.assertTrue(native.__file__.endswith(".so"))
        print("Engram FST:", native.__file__)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            for layer in (1, 14):
                expected = Engram.from_checkpoint(self.cfg, layer, folder, "cpu")
                scratch = self.load(folder, layer, "scratch")
                native_layer = self.load(folder, layer, "fastsafetensors")
                NewModelLoader._validate_loaded_weights(native_layer)
                self.assertEqual(len(list(native_layer.parameters())), 6)
                self.assertEqual(
                    native_layer.embed.weight.data_ptr(),
                    native_layer.embed_tokens.weight.data_ptr(),
                )
                for (key, value), (key2, reference) in zip(
                    native_layer.named_parameters(), scratch.named_parameters()
                ):
                    self.assertEqual(key, key2)
                    self.assertEqual(value.dtype, reference.dtype)
                    self.assertTrue(
                        torch.equal(
                            value.view(torch.uint8), reference.view(torch.uint8)
                        )
                    )
        gc.collect()
        churn = [torch.full((96, 32), i, dtype=torch.uint8) for i in range(32)]
        self.assertEqual(len(churn), 32)
        hidden = (torch.arange(3 * 2 * 32).reshape(3, 2, 32) % 13).bfloat16()
        ids = torch.tensor([[0], [7], [-1]])
        mask = torch.tensor([True, True, False])
        torch.testing.assert_close(
            native_layer(hidden, ids, mask), expected(hidden, ids, mask), atol=0, rtol=0
        )

    def test_nonlocal_stage_does_not_open_or_read_other_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            self.fixture(path)
            (path / "layer-14.safetensors").unlink()
            self.load(folder, 1, "fastsafetensors")
            with self.assertRaises(FileNotFoundError):
                self.load(folder, 14, "fastsafetensors")

    def test_missing_scale_or_incorrect_scale_format_not_silently_used(self):
        for modify in (
            lambda t: t.pop("embed.scale"),
            lambda t: t.update({"embed.scale": t["embed.scale"].view(torch.uint8)}),
            lambda t: t.update({"wkv.scale": t["wkv.scale"][:2]}),
        ):
            with tempfile.TemporaryDirectory() as folder:
                self.fixture(Path(folder), bad=modify)
                with self.assertRaises((ValueError, KeyError)):
                    self.load(folder, 1, "fastsafetensors")

    def test_must_receive_resolved_loader_no_implicit_fallback(self):
        with self.assertRaisesRegex(ValueError, "resolved"):
            self.load("unused", 1, "auto")


if __name__ == "__main__":
    unittest.main()
