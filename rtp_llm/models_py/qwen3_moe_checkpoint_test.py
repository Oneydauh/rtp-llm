"""Real CPU checkpoint -> production Qwen3-MoE parameter tree and service publication."""

import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file

from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3_moe.parameters import (
    install_qwen3_moe_parameters,
)
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.models_py.weight_mapper import WeightsMapper


def weights():
    # Real per-expert HF names and deliberately different values for every plane.
    shapes = {
        "model.embed_tokens.weight": (8, 4),
        "model.norm.weight": (4,),
        "lm_head.weight": (8, 4),
    }
    layer = "model.layers.0."
    for norm in ("input_layernorm", "post_attention_layernorm"):
        shapes[layer + norm + ".weight"] = (4,)
    for leaf, rows in (("q", 4), ("k", 2), ("v", 2), ("o", 4)):
        shapes[layer + "self_attn." + leaf + "_proj.weight"] = (rows, 4)
    for leaf in ("q", "k"):
        shapes[layer + "self_attn." + leaf + "_norm.weight"] = (2,)
    shapes[layer + "mlp.gate.weight"] = (4, 4)
    for i in range(4):
        for leaf in ("gate", "up", "down"):
            shapes[f"{layer}mlp.experts.{i}.{leaf}_proj.weight"] = (4, 4)
    return {
        key: torch.arange(torch.Size(shape).numel(), dtype=torch.float32).reshape(shape)
        + i
        for i, (key, shape) in enumerate(shapes.items())
    }


@register_model("_qwen3_moe_cpu_parameters")
class _CpuStage(RtpModule):
    """The native scheduler is not needed to exercise checkpoint parameter owners."""

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, model, loader):
        super().__init__()
        cfg = dict(
            model_config=model,
            num_layers=1,
            vocab_size=8,
            hidden_size=4,
            num_heads=2,
            num_kv_heads=1,
            head_dim=2,
            rms_norm_eps=1e-6,
            num_experts=4,
            top_k=2,
            moe_intermediate_size=4,
            moe_config=None,
            params_dtype=torch.float32,
            lm_head_params_dtype=torch.float32,
            quant_config=loader.quant_config,
            parallelism_config=loader.parallelism_config,
        )
        for name in ("attn_tp", "ffn_tp", "ep", "lm_head_tp"):
            for field in ("size", "rank"):
                key = f"{name}_{field}"
                cfg[key] = getattr(loader, key)
        install_qwen3_moe_parameters(self, cfg, layer_ids=range(1), partitioned=False)

    def load_weights(self, weights):
        return super().load_weights(self.WEIGHTS_MAPPER.apply(weights))

    def runtime_weight_view(self):
        return {"embedding": self.embed_tokens.weight, "lm_head": self.lm_head.weight}


class Qwen3MoeCpuTest(unittest.TestCase):
    def test_service_publication_scratch_and_fst(self):
        for method in ("scratch", "fastsafetensors"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
            ):
                model = service(directory, method, "_qwen3_moe_cpu_parameters")
                model.parallelism_config.local_world_size = 1
                cfg = dict(
                    hidden_size=4,
                    expert_num=4,
                    moe_k=2,
                    moe_inter_size=4,
                    moe_layer_index=[0],
                    n_shared_experts=0,
                    qk_norm=True,
                    scoring_func=0,
                    routed_scaling_factor=1.0,
                    has_moe_norm=True,
                    moe_n_group=1,
                    moe_topk_group=1,
                    activation_type="SiGLU",
                    tie_word_embeddings=False,
                    normalize_lm_head_weight=False,
                    logit_scale=1.0,
                )
                for key, value in cfg.items():
                    setattr(model.model_config, key, value)
                data = weights()
                save_file(data, str(Path(directory) / "model.safetensors"))
                load_owned_model(model)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                owner = model.py_model
                self.assertIs(
                    model.weight.get_global_weight("lm_head"), owner.lm_head.weight
                )
                self.assertTrue(
                    torch.equal(owner.lm_head.weight, data["lm_head.weight"])
                )
                dec = owner.layers[0]
                self.assertIsNone(dec.self_attn.qkv_proj.bias)
                self.assertTrue(
                    torch.equal(
                        dec.self_attn.qkv_proj.weight,
                        torch.cat(
                            [
                                data[f"model.layers.0.self_attn.{p}_proj.weight"]
                                for p in "qkv"
                            ]
                        ),
                    )
                )
                self.assertIsNone(dec.mlp.shared_experts)
                self.assertTrue(
                    torch.equal(
                        dec.self_attn.q_norm.weight,
                        data["model.layers.0.self_attn.q_norm.weight"],
                    )
                )
                self.assertTrue(
                    torch.equal(
                        dec.mlp.gate.weight, data["model.layers.0.mlp.gate.weight"]
                    )
                )
                for i in range(4):
                    prefix = f"model.layers.0.mlp.experts.{i}."
                    self.assertTrue(
                        torch.equal(
                            dec.mlp.experts.w13[i],
                            torch.cat(
                                [
                                    data[prefix + leaf + "_proj.weight"]
                                    for leaf in ("up", "gate")
                                ]
                            ),
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            dec.mlp.experts.w2[i], data[prefix + "down_proj.weight"]
                        )
                    )
                logits = torch.tensor([[1.0, 3.0, 2.0, -1.0]])
                self.assertTrue(dec.mlp.experts.is_cpu_checkpoint_staging())
                routes, ids = dec.mlp.route(logits, ids_dtype=torch.int32)
                torch.testing.assert_close(
                    routes, logits.gather(1, ids.long()).softmax(-1)
                )


if __name__ == "__main__":
    unittest.main()
