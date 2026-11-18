"""Main ABI + direct-core checkpoint regression; no GPU inference claim.

Run after the NewLoader core dependency is integrated. No native import stubs,
legacy weight mapping or fallback loader is allowed in this test.
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.models_py.registry import register_model

register_model("_test_main_pp_qwen3")(Qwen3ForCausalLM)


def model_config():
    return SimpleNamespace(
        model_type="_test_main_pp_qwen3",
        num_layers=3,
        is_mtp=False,
        vocab_size=8,
        hidden_size=4,
        inter_size=4,
        attn_config=SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=2),
        layernorm_eps=1e-6,
        enable_fp32_lm_head=False,
        tie_word_embeddings=True,
        capture_aux_hidden_layer_ids=[],
        hidden_state_capture_layer_ids=[],
    )


def checkpoint():
    values = {
        "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
            8, 4
        ),
        "model.norm.weight": torch.ones(4),
    }
    for layer in range(3):
        prefix = f"model.layers.{layer}."
        for name, shape in (
            ("input_layernorm.weight", (4,)),
            ("post_attention_layernorm.weight", (4,)),
            ("self_attn.q_norm.weight", (2,)),
            ("self_attn.k_norm.weight", (2,)),
            ("self_attn.q_proj.weight", (4, 4)),
            ("self_attn.k_proj.weight", (2, 4)),
            ("self_attn.v_proj.weight", (2, 4)),
            ("self_attn.o_proj.weight", (4, 4)),
            ("mlp.gate_proj.weight", (4, 4)),
            ("mlp.up_proj.weight", (4, 4)),
            ("mlp.down_proj.weight", (4, 4)),
        ):
            values[prefix + name] = torch.full(shape, layer + 1.0)
    return values


class MainPPCheckpointTest(unittest.TestCase):
    def test_pp1_keeps_full_owner_and_main_capture_capability(self):
        parallel = SimpleNamespace(
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            pp_size=1,
            pp_rank=0,
            pp_stage_layer_counts=[],
            dp_size=1,
            dp_rank=0,
            world_size=1,
            world_rank=0,
            get_attn_tp_size=lambda: 1,
            get_attn_tp_rank=lambda: 0,
            get_ffn_tp_size=lambda: 1,
            get_ffn_tp_rank=lambda: 0,
        )
        load = NewLoaderConfig(
            device="cpu",
            load_method="scratch",
            compute_dtype=torch.float32,
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallel,
        )
        with tempfile.TemporaryDirectory() as folder:
            save_file(checkpoint(), os.path.join(folder, "model.safetensors"))
            model = NewModelLoader(model_config(), load, model_path=folder).load()
        self.assertIsInstance(model.layers, torch.nn.ModuleList)
        self.assertEqual(len(model.layers), 3)
        self.assertIsNotNone(model.embed_tokens)
        self.assertIsNotNone(model.lm_head)
        self.assertTrue(model.supports_hidden_state_capture)
        torch.testing.assert_close(model.embed_tokens.weight, model.lm_head.weight)

    def test_stage_ownership_and_global_tied_head_rows(self):
        for stage in (0, 1):
            for tp, rank in ((1, 0), (2, 0), (2, 1)):
                for pruned in (False, True):
                    for explicit in (False, True):
                        with self.subTest(
                            stage=stage,
                            tp=tp,
                            rank=rank,
                            pruned=pruned,
                            explicit=explicit,
                        ):
                            config = model_config()
                            if pruned:
                                config.enable_output_vocab_pruning = True
                                config.output_vocab_ids = [1, 3, 7]
                                config.output_vocab_padded_size = 4
                            parallel = SimpleNamespace(
                                tp_size=tp,
                                tp_rank=rank,
                                ep_size=1,
                                ep_rank=0,
                                pp_size=2,
                                pp_rank=stage,
                                pp_stage_layer_counts=[1, 2],
                                dp_size=1,
                                dp_rank=0,
                                world_size=2 * tp,
                                world_rank=stage * tp + rank,
                                get_attn_tp_size=lambda: tp,
                                get_attn_tp_rank=lambda: rank,
                                get_ffn_tp_size=lambda: tp,
                                get_ffn_tp_rank=lambda: rank,
                            )
                            load = NewLoaderConfig(
                                tp_size=tp,
                                tp_rank=rank,
                                compute_dtype=torch.float32,
                                device="cpu",
                                load_method="scratch",
                                quant_config=QuantizationConfig("none"),
                                parallelism_config=parallel,
                            )
                            values = checkpoint()
                            if explicit:
                                values["lm_head.weight"] = (
                                    values["model.embed_tokens.weight"] + 17
                                )
                            with tempfile.TemporaryDirectory() as folder:
                                save_file(
                                    values, os.path.join(folder, "model.safetensors")
                                )
                                model = NewModelLoader(
                                    config, load, model_path=folder
                                ).load()
                            self.assertEqual(
                                list(model.layers), ["0"] if stage == 0 else ["1", "2"]
                            )
                            self.assertEqual(model.embed_tokens is not None, stage == 0)
                            self.assertEqual(model.lm_head is not None, stage == 1)
                            for layer_id, layer in model.layers.items():
                                torch.testing.assert_close(
                                    layer.input_layernorm.weight,
                                    torch.full((4,), int(layer_id) + 1.0),
                                )
                            template = torch.zeros(2, 4)
                            self.assertIs(
                                model.make_empty_intermediate_tensors(template)[
                                    "hidden_states"
                                ],
                                template,
                            )
                            if stage == 1:
                                expected = values[
                                    (
                                        "lm_head.weight"
                                        if explicit
                                        else "model.embed_tokens.weight"
                                    )
                                ]
                                if pruned:
                                    selected = torch.zeros(4, 4)
                                    selected[:3] = expected[[1, 3, 7]]
                                    expected = selected
                                torch.testing.assert_close(
                                    model.lm_head.weight, expected.chunk(tp)[rank]
                                )
                                self.assertEqual(
                                    model.lm_head._explicit_checkpoint_loaded, explicit
                                )


if __name__ == "__main__":
    unittest.main()
