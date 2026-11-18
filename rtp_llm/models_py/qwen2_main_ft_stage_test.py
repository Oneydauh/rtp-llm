"""Requires the matching main native bindings; no inference backend is stubbed."""

import tempfile
import unittest
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.tools.convert.ft_export_metadata import generate_ft_index
from rtp_llm.tools.convert.ft_export_writer import write_ft_shards


def produce_records(rank):
    hidden, local_ffn, qkv = 4, 2, 6
    rows = torch.arange(32, dtype=torch.float32).reshape(8, hidden)
    yield None, "embedding", rows.chunk(2, dim=1)[rank]
    padded = torch.cat([rows + 91, torch.zeros_like(rows)], dim=0)
    yield None, "lm_head", padded.chunk(2)[rank]
    yield None, "final_layernorm.gamma", torch.full((hidden,), 7.0)
    for layer_id in range(3):
        base = layer_id * 1000 + rank * 100
        for key, shape in [
            ("pre_layernorm_weights.gamma", (hidden,)),
            ("post_layernorm_weights.gamma", (hidden,)),
            ("self_attention_weights.query_weight.kernel", (hidden, qkv)),
            ("self_attention_weights.query_weight.bias", (1, qkv)),
            ("self_attention_weights.attention_output_weight.kernel", (2, hidden)),
            ("ffn_weights.intermediate_weight13.kernel", (hidden, 2 * local_ffn)),
            ("ffn_weights.intermediate_weight2.kernel", (local_ffn, hidden)),
        ]:
            yield layer_id, key, torch.arange(
                torch.Size(shape).numel(), dtype=torch.float32
            ).reshape(shape) + base


class MainFTStageTest(unittest.TestCase):
    def test_main_writer_to_qwen2_stage_owners(self):
        # Exercise main's actual serialization and indexing with deterministic
        # already-converted records, not a mock of FT or a full HF conversion.
        with tempfile.TemporaryDirectory() as directory:
            for rank in (0, 1):
                write_ft_shards(
                    produce_records(rank), directory, tp_rank=rank, ep_rank=rank
                )
            generate_ft_index(directory, {"TP_SIZE": 2, "DP_SIZE": 1, "EP_SIZE": 0})
            config = SimpleNamespace(
                model_type="qwen_2",
                num_layers=3,
                is_mtp=False,
                vocab_size=8,
                hidden_size=4,
                inter_size=4,
                attn_config=SimpleNamespace(head_num=2, kv_head_num=1, size_per_head=2),
                layernorm_eps=1e-6,
                enable_fp32_lm_head=False,
                tie_word_embeddings=False,
                capture_aux_hidden_layer_ids=[],
                hidden_state_capture_layer_ids=[],
            )
            for method in ("scratch", "fastsafetensors"):
                for stage in (0, 1):
                    for rank in (0, 1):
                        pc = SimpleNamespace(
                            tp_size=2,
                            tp_rank=rank,
                            ep_size=2,
                            ep_rank=rank,
                            pp_size=2,
                            pp_rank=stage,
                            pp_stage_layer_counts=[1, 2],
                            dp_size=1,
                            dp_rank=0,
                            world_size=4,
                            world_rank=2 * stage + rank,
                            get_attn_tp_size=lambda: 2,
                            get_attn_tp_rank=lambda: rank,
                            get_ffn_tp_size=lambda: 2,
                            get_ffn_tp_rank=lambda: rank,
                        )
                        load = NewLoaderConfig(
                            tp_size=2,
                            tp_rank=rank,
                            ep_size=2,
                            ep_rank=rank,
                            compute_dtype=torch.float32,
                            device="cpu",
                            load_method=method,
                            quant_config=QuantizationConfig("none"),
                            parallelism_config=pc,
                        )
                        loader = NewModelLoader(config, load, model_path=directory)
                        model = loader.load()
                        self.assertEqual(loader.loaded_method.value, method)
                        self.assertEqual(
                            tuple(model.layers), ("1", "2") if stage else ("0",)
                        )
                        expected = (
                            torch.arange(32, dtype=torch.float32)
                            .reshape(8, 4)
                            .chunk(2)[rank]
                        )
                        if stage:
                            self.assertIsNone(model.embed_tokens)
                            torch.testing.assert_close(
                                model.lm_head.weight, expected + 91
                            )
                        else:
                            self.assertIsNone(model.lm_head)
                            torch.testing.assert_close(
                                model.embed_tokens.weight, expected
                            )
                        for lid, module in model.layers.items():
                            record = {
                                k: v
                                for i, k, v in produce_records(rank)
                                if i == int(lid)
                            }
                            torch.testing.assert_close(
                                module.input_layernorm.weight,
                                record["pre_layernorm_weights.gamma"],
                            )
                            torch.testing.assert_close(
                                module.self_attn.qkv_proj.weight,
                                record["self_attention_weights.query_weight.kernel"].T,
                            )


if __name__ == "__main__":
    unittest.main()
