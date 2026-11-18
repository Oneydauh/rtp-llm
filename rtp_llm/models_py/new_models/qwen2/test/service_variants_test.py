"""Exercise the real service loader and output contracts, without a GPU kernel."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from rtp_llm.config.generate_config import GenerateConfig
from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.cpp.model_rpc.model_rpc_client import StreamState, trans_output
from rtp_llm.cpp.model_rpc.proto.model_rpc_service_pb2 import (
    GenerateOutputsPB,
    TensorPB,
)
from rtp_llm.model_factory import ModelFactory
from rtp_llm.models.base_model import BaseModel
from rtp_llm.ops import ParallelismConfig, TaskType
from rtp_llm.utils.base_model_datatypes import GenerateInput


def fixture(root, *, head):
    raw = dict(
        architectures=["Qwen2ForCausalLM"],
        model_type="qwen2",
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=8,
        vocab_size=8,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
    )
    (root / "config.json").write_text(json.dumps(raw))
    tokenizer = Tokenizer(WordLevel({str(i): i for i in range(8)}, unk_token="0"))
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="0", eos_token="7"
    ).save_pretrained(root)
    values = {
        "model.embed_tokens.weight": torch.eye(8, 4),
        "model.norm.weight": torch.ones(4),
    }
    if head:
        values["lm_head.weight"] = torch.arange(32).reshape(8, 4).float()
    prefix = "model.layers.0."
    for role in ("input_layernorm", "post_attention_layernorm"):
        values[prefix + role + ".weight"] = torch.ones(4)
    for role, width in (("q_proj", 4), ("k_proj", 2), ("v_proj", 2)):
        values[prefix + "self_attn." + role + ".weight"] = torch.eye(width, 4)
        values[prefix + "self_attn." + role + ".bias"] = torch.zeros(width)
    values[prefix + "self_attn.o_proj.weight"] = torch.eye(4)
    for role, rows, cols in (
        ("gate_proj", 8, 4),
        ("up_proj", 8, 4),
        ("down_proj", 4, 8),
    ):
        values[prefix + "mlp." + role + ".weight"] = torch.eye(rows, cols)
    shards = {}
    for index, (key, tensor) in enumerate(values.items()):
        shard_name = f"model-{index % 2}.safetensors"
        shards.setdefault(shard_name, {})[key] = tensor
    weight_map = {}
    for shard_name, tensors in shards.items():
        save_file(tensors, str(root / shard_name))
        weight_map.update(dict.fromkeys(tensors, shard_name))
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    return values


def load(root, name, method):
    cfg = ModelFactory.get_model_cls(name).create_config(str(root))
    cfg.model_type, cfg.tokenizer_path, cfg.data_type = name, str(root), "fp32"
    cfg.task_type = (
        TaskType.DENSE_EMBEDDING
        if name == "qwen_2_embedding"
        else TaskType.LANGUAGE_MODEL
    )
    cfg.use_kvcache = cfg.task_type == TaskType.LANGUAGE_MODEL
    cfg.render_config, cfg.generate_env_config = RenderConfig(), GenerateEnvConfig()
    ns = SimpleNamespace
    engine = ns(
        parallelism_config=ParallelismConfig(),
        runtime_config=ns(
            warm_up=False, model_warm_up=False, max_generate_batch_size=4
        ),
        hw_kernel_config=ns(enable_cuda_graph=False, use_swizzleA=False),
        kv_cache_config=ns(multi_task_prompt="", multi_task_prompt_str=""),
        fmha_config=None,
        moe_config=None,
        load_config=ns(
            load_method=method,
            force_cpu_load_weights=False,
            loader_recycle_handles=False,
            moe_pure_tp_preshard=False,
        ),
    )
    with patch.object(BaseModel, "_get_device_str", return_value="cpu"):
        result = ModelFactory._create_model(cfg, engine)
    assert result.model_weights_loader.loaded_method.value == method
    return result


class Qwen2ServiceVariantsTest(unittest.TestCase):
    def test_tool_selected_logits_survive_terminal_frame(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tensors = fixture(root, head=True)
                svc = load(root, "qwen_tool", method)
                self.assertTrue(svc.py_model.config.attn_config.is_causal)
                self.assertIsNone(svc.custom_module)
                weight = svc.py_model.runtime_weight_view()["lm_head"]
                torch.testing.assert_close(weight, tensors["lm_head.weight"])
                request = GenerateInput(
                    request_id=1,
                    token_ids=torch.tensor([0]),
                    mm_inputs=[],
                    generate_config=GenerateConfig(
                        return_logits=True,
                        logits_index=2,
                        select_tokens_id=[1, 2],
                        aux_info=True,
                        is_streaming=True,
                    ),
                )
                stream = StreamState()
                for step in (1, 2, 3):
                    logits = weight[[1, 2]] @ torch.tensor([float(step), 0, 0, 0])
                    response = GenerateOutputsPB()
                    flat = response.flatten_output
                    flat.finished.append(step == 3)
                    flat.aux_info.add().output_len = step
                    flat.logits.data_type = TensorPB.FP32
                    flat.logits.shape.extend([1, 1, 2])
                    flat.logits.fp32_data = logits.numpy().tobytes()
                    output = trans_output(request, response, stream).generate_outputs[0]
                self.assertTrue(output.finished)
                torch.testing.assert_close(output.logits, torch.tensor([[8.0, 16.0]]))

    def test_embedding_publishes_noncausal_backbone_and_pooling_not_a_head(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=False)
                svc = load(root, "qwen_2_embedding", method)
                owner = svc.py_model
                self.assertFalse(owner.config.attn_config.is_causal)
                self.assertIsNone(owner.lm_head)
                self.assertNotIn("lm_head", owner.runtime_weight_view())
                torch.testing.assert_close(owner.norm.weight, torch.ones(4))
                # Two packed sequences; bidirectional NormalHandler pools the
                # first token, not the last. Use the weights actually loaded.
                ids = torch.tensor([1, 3, 2, 0, 1], dtype=torch.int32)
                hidden = owner.embed_tokens.weight[ids] * 3
                result = svc.custom_module.get_handler().forward(
                    ids, hidden, torch.tensor([2, 3])
                )
                torch.testing.assert_close(result, torch.eye(4)[[1, 2]])

    def test_embedding_respects_checkpoint_pooling_program(self):
        # A sentence-transformers release may explicitly select the last
        # token even with fully bidirectional attention in the transformer.
        from sentence_transformers.models import Pooling

        for method in ("scratch", "fastsafetensors"):
            with self.subTest(
                method=method
            ), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root, head=False)
                pool_path = root / "1_Pooling"
                pool_path.mkdir()
                Pooling(4, pooling_mode="lasttoken").save(str(pool_path))
                (root / "modules.json").write_text(
                    json.dumps(
                        [
                            {
                                "idx": 0,
                                "name": "0",
                                "path": "",
                                "type": "sentence_transformers.models.Transformer",
                            },
                            {
                                "idx": 1,
                                "name": "1",
                                "path": "1_Pooling",
                                "type": "sentence_transformers.models.Pooling",
                            },
                        ]
                    )
                )
                svc = load(root, "qwen_2_embedding", method)
                self.assertFalse(svc.py_model.config.attn_config.is_causal)
                ids = torch.tensor([1, 3, 2, 0, 1], dtype=torch.int32)
                hidden = svc.py_model.embed_tokens.weight[ids]
                handler = svc.custom_module.get_handler()
                result = handler.extend_forward(
                    input_ids=ids,
                    hidden_states=hidden,
                    input_lengths=torch.tensor([2, 3]),
                    attention_mask=torch.ones_like(ids),
                )
                torch.testing.assert_close(result, torch.eye(4)[[3, 1]])


if __name__ == "__main__":
    unittest.main()
