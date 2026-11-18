"""Actual BERT projection checkpoints versus the main ROCm hipBLASLt executor."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file

from rtp_llm.models_py.new_models.bert.parameters import BertLayer
from rtp_llm.models_py.quant_methods.base import QuantizationConfig


class BertPtpcGpuEpilogueTest(unittest.TestCase):
    def test_bert_projection_checkpoint_bias_and_gelu_match_main(self):
        self.assertIsNotNone(torch.version.hip, "requires the ROCm target")
        self.assertTrue(torch.cuda.is_available(), "requires the assigned GPU")
        from rtp_llm.models_py.modules.factory.linear.impl.rocm.fp8_ptpc_linear import (
            RocmFp8PTPCLinearWithSwizzle,
        )

        cfg = SimpleNamespace(
            hidden_size=128,
            inter_size=256,
            layernorm_eps=1e-5,
            attn_config=SimpleNamespace(head_num=4, size_per_head=32),
        )
        load = SimpleNamespace(
            attn_tp_size=1,
            attn_tp_rank=0,
            ffn_tp_size=1,
            ffn_tp_rank=0,
            compute_dtype=torch.bfloat16,
            quant_config=QuantizationConfig(
                "fp8_per_channel", hw_kernel_config=SimpleNamespace(use_swizzleA=True)
            ),
        )
        layer = BertLayer(cfg, load, 0).cuda()
        state = {}
        torch.manual_seed(91296857)
        for name in ("qkv", "attn_output", "intermediate", "output"):
            projection = getattr(layer, name)
            state[name + ".weight"] = (torch.randn(projection.weight.shape) * 0.3).to(
                torch.float8_e4m3fn
            )
            state[name + ".weight_scale"] = torch.linspace(
                0.07, 0.19, projection.weight.shape[0]
            ).reshape(-1, 1)
            state[name + ".bias"] = (torch.randn(projection.bias.shape) * 0.2).to(
                torch.bfloat16
            )
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "bert_layer.safetensors")
            save_file(state, path)
            checkpoint = load_file(path)
            for name in ("qkv", "attn_output", "intermediate", "output"):
                with self.subTest(projection=name):
                    projection = getattr(layer, name)
                    projection.load_weights(
                        {
                            key[len(name) + 1 :]: value
                            for key, value in checkpoint.items()
                            if key.startswith(name + ".")
                        }
                    )
                    projection.process_weights_after_loading()
                    self.assertEqual(
                        projection.quant_method._runtime_backend, "rocm_hipblaslt_ptpc"
                    )
                    # Exactly the legacy LinearFactory executor and epilogue,
                    # using the same final weight/scale storage, not a fake GEMM.
                    main = RocmFp8PTPCLinearWithSwizzle(
                        weight=projection.weight,
                        weight_scales=projection.weight_scale,
                        bias=projection.bias,
                    )
                    for batch in (1, 7, 128):
                        x = torch.randn(
                            batch,
                            projection.input_size,
                            dtype=torch.bfloat16,
                            device="cuda",
                        )
                        if name == "intermediate":
                            expected = main.forward_with_bias_gelu(x)
                            actual = projection.forward_with_bias_gelu(x)
                        else:
                            expected = main(x)
                            actual = projection(x)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
