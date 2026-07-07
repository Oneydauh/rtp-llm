"""Load completeness and input-layout regression tests for newloader."""

import types
import unittest
from unittest import mock

import torch

from rtp_llm.models_py.layers.embedding import VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import LayerNorm, RMSNorm
from rtp_llm.models_py.layers.conv import Conv3dLayer
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.fp8 import Fp8LinearMethod, _runtime_fp8_dtype

# Register MoE quant methods for BaseMoEExperts tests.
import rtp_llm.models_py.quant_methods.fp8_moe  # noqa: F401


def _qc(quant_type: str) -> QuantizationConfig:
    return QuantizationConfig(quant_type=quant_type)


class TestWeightMapperStreaming(unittest.TestCase):
    def test_safetensors_loader_reads_one_tensor_at_a_time(self):
        class FakeSafeOpen:
            def __enter__(self):
                self.read_names = []
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def keys(self):
                return ["a.weight", "b.weight"]

            def get_tensor(self, name):
                self.read_names.append(name)
                return torch.full((1,), len(self.read_names), dtype=torch.float32)

        fake = FakeSafeOpen()

        def fake_safe_open(path, framework, device):
            self.assertEqual(path, "model.safetensors")
            self.assertEqual(framework, "pt")
            self.assertEqual(device, "cpu")
            return fake

        with mock.patch("safetensors.safe_open", side_effect=fake_safe_open):
            items = list(
                weight_mapper.get_all_weights(["model.safetensors"], device="cpu")
            )

        self.assertEqual([name for name, _ in items], ["a.weight", "b.weight"])
        self.assertEqual(fake.read_names, ["a.weight", "b.weight"])


class TestRtpModuleDispatchIntegrity(unittest.TestCase):
    class Container(RtpModule):
        def __init__(self):
            super().__init__()
            self.child = torch.nn.Linear(2, 2, bias=False)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)])

    def test_nested_missing_leaf_fails(self):
        module = self.Container()
        with self.assertRaisesRegex(RuntimeError, "child.missing"):
            module.load_weights({"child.missing": torch.zeros(2, 2)})

    def test_module_list_bad_index_fails(self):
        module = self.Container()
        with self.assertRaisesRegex(RuntimeError, "layers.3.weight"):
            module.load_weights({"layers.3.weight": torch.zeros(2, 2)})


class TestLinearLoadCompleteness(unittest.TestCase):
    def test_column_missing_weight_fails_before_post_load(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_qc("none"),
            prefix="linear",
            params_dtype=torch.float32,
        )
        with self.assertRaisesRegex(RuntimeError, "missing required checkpoint tensors"):
            layer.process_weights_after_loading()

    def test_fp8_missing_scale_fails(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=4,
            quant_config=_qc("fp8"),
            prefix="fp8_linear",
            params_dtype=torch.float32,
        )
        layer.load_weights({"fp8_linear.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype())})
        with self.assertRaisesRegex(RuntimeError, "weight_scale"):
            layer.process_weights_after_loading()

    def test_merged_missing_shard_fails(self):
        layer = MergedColumnParallelLinear(
            input_size=4,
            output_size=8,
            quant_config=_qc("none"),
            prefix="gate_up_proj",
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
        )
        layer.load_weights({"gate_up_proj.gate_proj.weight": torch.zeros(4, 4)})
        with self.assertRaisesRegex(RuntimeError, "up_proj|1"):
            layer.process_weights_after_loading()

    def test_qkv_missing_v_shard_fails(self):
        layer = QKVParallelLinear(
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            quant_config=_qc("none"),
            prefix="qkv_proj",
            params_dtype=torch.float32,
        )
        layer.load_weights(
            {
                "qkv_proj.q_proj.weight": torch.zeros(4, 4),
                "qkv_proj.k_proj.weight": torch.zeros(2, 4),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "v"):
            layer.process_weights_after_loading()


class TestMoELoadCompleteness(unittest.TestCase):
    def _make_moe(self):
        return BaseMoEExperts(
            num_experts=1,
            hidden_size=4,
            moe_intermediate_size=4,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            params_dtype=torch.float32,
            model_config=types.SimpleNamespace(
                data_type="fp32", quant_config=None, exported_device=None
            ),
            parallelism_config=types.SimpleNamespace(dp_size=1),
            moe_config=types.SimpleNamespace(),
            quant_config=_qc("fp8"),
            layer_idx=0,
        )

    def test_missing_quant_scale_fails(self):
        layer = self._make_moe()
        layer.load_weights(
            {
                "0.gate_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
                "0.up_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
                "0.down_proj.weight": torch.zeros(4, 4, dtype=_runtime_fp8_dtype()),
            }
        )
        with self.assertRaisesRegex(RuntimeError, "auxiliary tensors"):
            layer.process_weights_after_loading()


class TestOtherLayerLoadCompleteness(unittest.TestCase):
    def test_embedding_missing_weight_fails(self):
        layer = VocabParallelEmbedding(8, 4, params_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "weight"):
            layer.process_weights_after_loading()

    def test_norm_missing_weight_fails(self):
        layer = RMSNorm(4, params_dtype=torch.float32)
        with self.assertRaisesRegex(RuntimeError, "weight"):
            layer.process_weights_after_loading()

    def test_layernorm_missing_bias_fails(self):
        layer = LayerNorm(4, params_dtype=torch.float32)
        layer.load_weights({"weight": torch.ones(4)})
        with self.assertRaisesRegex(RuntimeError, "bias"):
            layer.process_weights_after_loading()

    def test_conv_missing_bias_fails(self):
        layer = Conv3dLayer(1, 1, (1, 1, 1), params_dtype=torch.float32)
        layer.load_weights({"weight": torch.ones_like(layer.conv.weight)})
        with self.assertRaisesRegex(RuntimeError, "bias"):
            layer.process_weights_after_loading()


class TestFp8ForwardInputLayout(unittest.TestCase):
    def test_fp8_per_tensor_accepts_non_contiguous_input(self):
        layer = ColumnParallelLinear(
            input_size=4,
            output_size=3,
            quant_config=_qc("fp8"),
            prefix="fp8_linear",
            params_dtype=torch.float32,
        )
        layer.load_weights(
            {
                "fp8_linear.weight": torch.zeros(3, 4, dtype=_runtime_fp8_dtype()),
                "fp8_linear.weight_scale": torch.tensor([1.0], dtype=torch.float32),
            }
        )
        layer.process_weights_after_loading()
        x = torch.randn(2, 4, 3).transpose(1, 2)
        self.assertFalse(x.is_contiguous())

        def fake_quant(inp):
            self.assertEqual(inp.shape, (6, 4))
            self.assertTrue(inp.is_contiguous())
            return inp.to(_runtime_fp8_dtype()), torch.ones(1, dtype=torch.float32)

        def fake_scaled_mm(a, b, **kwargs):
            return torch.zeros(a.shape[0], b.shape[1], dtype=torch.float32)

        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8._resolve_per_tensor_quant",
            return_value=fake_quant,
        ), mock.patch.object(torch, "_scaled_mm", side_effect=fake_scaled_mm):
            out = layer(x)
        self.assertEqual(out.shape, (2, 3, 3))


if __name__ == "__main__":
    unittest.main()
