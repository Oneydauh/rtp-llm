"""Dense and sparse DeepSeek-V3 family reach the public direct serving owner."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.deepseek_v2 import DeepSeekV2
from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.deepseek_v3.language import DeepSeekV32ForCausalLM
from rtp_llm.ops import ParallelismConfig
from rtp_llm.utils.model_weight import W


def _deterministic_values(shape, offset):
    count = math.prod(shape)
    return (
        torch.arange(offset, offset + count, dtype=torch.float32).reshape(shape)
        / max(count, 1)
        - 0.5
    )


def _score_weights():
    values = _deterministic_values
    weights = {
        "model.embed_tokens.weight": values((32, 32), 0),
        "model.norm.weight": values((32,), 3) + 1.0,
        "lm_head.weight": values((32, 32), 5),
    }
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}."
        offset = 11 + layer_idx * 100
        weights.update(
            {
                prefix + "input_layernorm.weight": values((32,), offset) + 1.0,
                prefix + "self_attn.q_a_proj.weight": values((32, 32), offset + 3),
                prefix
                + "self_attn.q_a_layernorm.weight": values((32,), offset + 5)
                + 1.0,
                prefix + "self_attn.q_b_proj.weight": values((32, 32), offset + 7),
                prefix
                + "self_attn.kv_a_proj_with_mqa.weight": values((36, 32), offset + 11),
                prefix
                + "self_attn.kv_a_layernorm.weight": values((32,), offset + 13)
                + 1.0,
                prefix + "self_attn.kv_b_proj.weight": values((32, 32), offset + 17),
                prefix + "self_attn.o_proj.weight": values((32, 16), offset + 19),
                prefix
                + "post_attention_layernorm.weight": values((32,), offset + 23)
                + 1.0,
            }
        )
    weights.update(
        {
            "model.layers.0.mlp.gate_proj.weight": values((32, 32), 31),
            "model.layers.0.mlp.up_proj.weight": values((32, 32), 37),
            "model.layers.0.mlp.down_proj.weight": values((32, 32), 41),
            "model.layers.1.mlp.gate.weight": values((4, 32), 131),
            "model.layers.1.mlp.gate.e_score_correction_bias": values((4,), 137),
            "model.layers.1.mlp.shared_experts.gate_proj.weight": values((32, 32), 139),
            "model.layers.1.mlp.shared_experts.up_proj.weight": values((32, 32), 149),
            "model.layers.1.mlp.shared_experts.down_proj.weight": values((32, 32), 151),
        }
    )
    for expert_id in range(4):
        prefix = f"model.layers.1.mlp.experts.{expert_id}."
        offset = 157 + expert_id * 17
        weights[prefix + "gate_proj.weight"] = values((32, 32), offset)
        weights[prefix + "up_proj.weight"] = values((32, 32), offset + 3)
        weights[prefix + "down_proj.weight"] = values((32, 32), offset + 5)
    return weights



class ScoreRouteTest(unittest.TestCase):
    def test_main_config_load_and_sparse_runtime_ownership(self):
        for tag, hf, sparse, interleave in (
            ("deepseek3", "deepseek_v3", False, False),
            ("deepseek_v32", "deepseek_v32", True, False),
            ("glm_5", "glm_moe_dsa", True, True),
        ):
            data = _score_weights()
            cfg = dict(
                model_type=hf, hidden_size=32, intermediate_size=32,
                max_position_embeddings=128, moe_intermediate_size=32, topk_method="noaux_tc",
                scoring_func="sigmoid", norm_topk_prob=True, num_attention_heads=4, n_group=1,
                topk_group=1, n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
                num_hidden_layers=2, num_nextn_predict_layers=1, first_k_dense_replace=1,
                moe_layer_freq=1, q_lora_rank=32, kv_lora_rank=32, qk_nope_head_dim=4,
                qk_rope_head_dim=4, v_head_dim=4, vocab_size=32, routed_scaling_factor=2.5,
                rms_norm_eps=1e-6, rope_theta=10000, torch_dtype="bfloat16",
            )
            if sparse:
                cfg.update(index_n_heads=2, index_head_dim=128, index_topk=4, indexer_rope_interleave=interleave)
                for layer in range(2):
                    prefix = f"model.layers.{layer}.self_attn.indexer."
                    for key, shape in {
                        "wq_b.weight": (256, 32), "wk.weight": (128, 32),
                        "k_norm.weight": (128,), "k_norm.bias": (128,),
                        "weights_proj.weight": (2, 32),
                    }.items():
                        data[prefix + key] = _deterministic_values(shape, 17 + layer)
            data["model.layers.2.embed_tokens.weight"] = data["model.embed_tokens.weight"].clone()
            with tempfile.TemporaryDirectory() as root:
                (Path(root) / "config.json").write_text(json.dumps(cfg))
                save_file(data, str(Path(root) / "model.safetensors"))
                config = DeepSeekV2._create_config(root)
                config.model_type, config.ckpt_path, config.max_seq_len = tag, root, 128
                # Main fills attention page geometry after parsing HF config.
                config.attn_config.tokens_per_block = 64
                config.attn_config.kernel_tokens_per_block = 64
                DeepSeekV2._post_build_model_config(config)
                config.data_type, config.enable_fp32_lm_head, config.lora_infos = "bf16", False, {}
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(model=tag, method=method):
                        service = SimpleNamespace(
                            model_config=config, _weight_alias_names=(), _weight_alias_owner=None,
                            _get_device_str=lambda: "cpu", parallelism_config=ParallelismConfig(),
                            hw_kernel_config=None, moe_config=None, fmha_config=None,
                            max_generate_batch_size=1, load_method=method, custom_module=None,
                        )
                        load_owned_model(service)
                        model = service.py_model
                        self.assertIsInstance(model, DeepSeekV32ForCausalLM)
                        self.assertEqual(service.model_weights_loader.loaded_method, NewLoaderLoadMethod(method))
                        self.assertTrue(model.initialize(SimpleNamespace(kv_cache=None)))
                        self.assertIs(model.weight.weights[1][W.mla_kc], model.layers[1].self_attn._kc_w)
                        self.assertEqual(model._get_fmha_group_tags(), ["default", "indexer_kv"] if sparse else None)
                        if sparse:
                            idx = model.layers[1].indexer
                            self.assertIs(idx.indexer_op.cos_sin_cache, model.cos_sin_cache)
                            self.assertEqual(idx.indexer_op.is_neox_style, not interleave)
                            torch.testing.assert_close(
                                idx.weights_proj.weight, data["model.layers.1.self_attn.indexer.weights_proj.weight"],
                            )
                            model.to(dtype=torch.bfloat16)
                            x = torch.ones(1, 32, dtype=torch.bfloat16)
                            scales = torch.ones(1, 2, 1)
                            expected = torch.nn.functional.linear(
                                x.float(), data["model.layers.1.self_attn.indexer.weights_proj.weight"]
                            ).unsqueeze(-1) * (idx.softmax_scale * idx.weights_scale)
                            torch.testing.assert_close(idx._get_logits_head_gate(x, scales), expected)
                        else:
                            self.assertIsNone(model.layers[1].indexer)


if __name__ == "__main__":
    unittest.main()
