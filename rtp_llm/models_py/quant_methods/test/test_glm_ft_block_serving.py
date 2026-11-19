"""H20 acceptance: main WeightConverter -> public model loading -> actual kernels.

Run on an allocated GPU with matching main native bindings. A CPU staging test
must never satisfy this target. It exercises projections/routing, not token smoke.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.quant_config import (
    Fp8BlockWiseQuantConfig,
    Fp8PerChannelCompressedQuantConfig,
)
from rtp_llm.models.glm4_moe import Glm4Moe
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.test.test_glm_ft_block_tree import (
    inputs,
    channel_inputs,
)
from rtp_llm.ops import FMHAConfig, HWKernelConfig, MoeConfig
from rtp_llm.tools.convert.weights_convert import WeightConverter


class LegacyWriterToServingTest(unittest.TestCase):
    def test_main_glm_quantized_writer_to_gpu(self):
        self._real_writer("block")

    def test_main_glm_channel_writer_to_gpu(self):
        self._real_writer("channel")

    def test_main_glm_online_block_writer_to_gpu(self):
        self._real_writer("online_block")

    def _real_writer(self, mode):
        self.assertTrue(
            torch.cuda.is_available(), "allocate an H20; CPU is not a substitute"
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hf, ft = root / "hf", root / "ft"
            hf.mkdir()
            ft.mkdir()
            # The real export descriptor initializes the production tokenizer.
            # Keep a complete, offline checkpoint with the same vocabulary as the head.
            tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer(
                    WordLevel(
                        {"[UNK]": 0, **{str(i): i for i in range(1, 8)}},
                        unk_token="[UNK]",
                    )
                ),
                unk_token="[UNK]",
            )
            tokenizer.save_pretrained(hf)
            (hf / "config.json").write_text(
                json.dumps(
                    dict(
                        architectures=["Glm4MoeForCausalLM"],
                        model_type="glm4_moe",
                        hidden_size=128,
                        intermediate_size=128,
                        moe_intermediate_size=128,
                        num_hidden_layers=1,
                        num_attention_heads=1,
                        num_key_value_heads=1,
                        head_dim=128,
                        vocab_size=8,
                        num_experts_per_tok=1,
                        n_routed_experts=2,
                        n_shared_experts=1,
                        routed_scaling_factor=1.0,
                        norm_topk_prob=True,
                        first_k_dense_replace=0,
                        n_group=1,
                        topk_group=1,
                        use_qk_norm=True,
                        attention_bias=True,
                        tie_word_embeddings=False,
                        torch_dtype="bfloat16",
                        max_position_embeddings=128,
                        quantization_config={
                            "quant_method": "fp8",
                            "fmt": "e4m3",
                            "activation_scheme": "dynamic",
                            "weight_block_size": [128, 128],
                        },
                    )
                )
            )
            weights, _ = inputs()
            config_path = hf / "config.json"
            metadata = json.loads(config_path.read_text())
            if mode == "channel":
                weights, _ = channel_inputs()
                metadata["quantization_config"] = {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "group_0": {
                            "weights": {
                                "num_bits": 8,
                                "strategy": "channel",
                                "type": "float",
                            },
                            "input_activations": None,
                        }
                    },
                }
            elif mode == "online_block":
                del metadata["quantization_config"]
                for key in list(weights):
                    if key.endswith(".weight_scale_inv"):
                        scale = weights.pop(key)
                        kernel = key.removesuffix("_scale_inv")
                        weights[kernel] = (weights[kernel].float() * scale).bfloat16()
            config_path.write_text(json.dumps(metadata))
            save_file(
                {
                    (k if k == "lm_head.weight" else "model." + k): v
                    for k, v in weights.items()
                },
                str(hf / "model.safetensors"),
            )
            env = dict(
                TP_SIZE="1", EP_SIZE="1", DP_SIZE="1", WORLD_SIZE="1", ACT_TYPE="bf16"
            )
            env["QUANTIZATION"] = "FP8_PER_BLOCK" if mode == "online_block" else ""
            converter = WeightConverter(str(hf), "glm4_moe", env)
            self.assertEqual(converter.convert(str(ft)), 0)
            self.assertIs(
                json.loads((ft / "model.safetensors.index.json").read_text())[
                    "is_ft_style_weight"
                ],
                True,
            )

            def load_and_evaluate(path, backend):
                cfg = Glm4Moe._create_config(str(path))
                cfg.model_type = "glm4_moe"
                cfg.ckpt_path = str(path)
                cfg.data_type = "bf16"
                cfg.enable_fp32_lm_head = False
                if mode == "channel":
                    source = Fp8PerChannelCompressedQuantConfig(is_quanted=True)
                    key = "fp8_per_channel"
                else:
                    source = Fp8BlockWiseQuantConfig(
                        bits=8, group_size=128, is_quanted=mode != "online_block"
                    )
                    key = "fp8_block_online" if mode == "online_block" else "fp8_block"
                cfg.quant_config = source
                config = NewLoaderConfig(
                    device="cuda:0",
                    compute_dtype=torch.bfloat16,
                    load_method=backend,
                    quant_config=QuantizationConfig(key, source),
                    parallelism_config=converter._build_parallelism_config(),
                    fmha_config=FMHAConfig(),
                    moe_config=MoeConfig(),
                    py_hw_kernel_config=HWKernelConfig(),
                )
                loader = NewModelLoader(cfg, config, model_path=str(path))
                model = loader.load()
                self.assertEqual(loader._selected_load_method.value, backend)
                layer = model.layers[0]
                experts = layer.mlp.experts
                self.assertIsNotNone(
                    experts.fused_moe, "load did not install a serving expert executor"
                )
                x = (torch.arange(4 * 128, device="cuda").reshape(4, 128) % 17 - 8).to(
                    torch.bfloat16
                ) / 128
                values = [
                    layer.self_attn.qkv_proj(x),
                    layer.self_attn.o_proj(x),
                    layer.mlp(x),
                ]
                values.extend(
                    [experts.w13, experts.w2, experts.w13_scale, experts.w2_scale]
                )
                torch.cuda.synchronize()
                return [
                    (
                        t.detach().float().cpu()
                        if t.dtype != torch.int32
                        else t.detach().cpu()
                    )
                    for t in values
                ]

            with torch.inference_mode():
                ref = load_and_evaluate(hf, "scratch")
                for method in ("scratch", "fastsafetensors"):
                    out = load_and_evaluate(ft, method)
                    for i, (expected, actual) in enumerate(zip(ref, out)):
                        torch.testing.assert_close(
                            expected,
                            actual,
                            rtol=0,
                            atol=0,
                            msg=f"{method} output/tensor {i}",
                        )


if __name__ == "__main__":
    unittest.main()
