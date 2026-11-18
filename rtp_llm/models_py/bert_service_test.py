"""Actual HF checkpoints -> public Bert service -> packed encoder/downstream outputs."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertForSequenceClassification, BertModel

from rtp_llm.config.py_config_modules import GenerateEnvConfig, RenderConfig
from rtp_llm.models.bert import Bert
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.bert.language import BertEncoderOwner
from rtp_llm.models_py.serving_route_checkpoint_test import service
from rtp_llm.ops import TaskType
from rtp_llm.utils.model_weight import W


class _CpuBert(Bert):
    def _get_device_str(self):
        return "cpu"


def make_checkpoint(root, *, classifier=False):
    cfg = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=20,
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=24,
        type_vocab_size=2,
        num_labels=3,
    )
    torch.manual_seed(1405)
    ref = (BertForSequenceClassification(cfg) if classifier else BertModel(cfg)).eval()
    cfg.save_pretrained(root)
    names = ref.state_dict()
    save_file(
        {k: v.contiguous() for k, v in names.items()}, str(root / "model.safetensors")
    )
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + [
        f"word{i}" for i in range(27)
    ]
    (root / "vocab.txt").write_text("\n".join(tokens) + "\n")
    BertWordPieceTokenizer(str(root / "vocab.txt")).save(str(root / "tokenizer.json"))
    (root / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "BertTokenizer"})
    )
    return ref


def public_load(root, method, task, *, tp=1, rank=0):
    config = Bert._create_config(str(root))
    config.model_type = "bert"
    config.tokenizer_path = str(root)
    config.data_type = "fp32"
    config.task_type = task
    config.generate_env_config = GenerateEnvConfig()
    config.render_config = RenderConfig()
    config.max_seq_len = (
        10  # A deployment cap is not the learned 24-row position table.
    )
    fixture = service(str(root), method)
    parallel = fixture.parallelism_config
    parallel.tp_size = parallel.world_size = tp
    parallel.tp_rank = parallel.local_rank = parallel.world_rank = rank
    parallel.get_attn_tp_size = parallel.get_ffn_tp_size = lambda: tp
    parallel.get_attn_tp_rank = parallel.get_ffn_tp_rank = lambda: rank
    parallel.pp_stage_layer_counts = []
    return _CpuBert.from_config(
        model_config=config,
        parallelism_config=parallel,
        kv_cache_config=SimpleNamespace(multi_task_prompt="", multi_task_prompt_str=""),
        hw_kernel_config=SimpleNamespace(enable_cuda_graph=False, use_swizzleA=False),
        moe_config=None,
        fmha_config=None,
        max_generate_batch_size=2,
        load_method=method,
        vit_config=None,
        merge_lora=False,
    )


def packed_inputs():
    ids = torch.tensor([2, 11, 3, 2, 15, 7, 3])
    lengths = torch.tensor([3, 4], dtype=torch.int32)
    pos = torch.tensor([0, 1, 2, 0, 1, 2, 3])
    types = torch.tensor([0, 1, 0, 0, 0, 1, 1])
    inputs = SimpleNamespace(
        input_ids=ids,
        attention_inputs=SimpleNamespace(input_lengths=lengths),
        bert_embedding_inputs=SimpleNamespace(
            combo_position_ids=pos,
            combo_tokens_type_ids=types,
            input_embedding_scalar=1.0,
        ),
    )
    return inputs, lengths


def check_tp_rank(rank, checkpoint, rendezvous):
    dist = torch.distributed
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        root = Path(checkpoint)
        state = load_file(str(root / "model.safetensors"))
        for method in ("scratch", "fastsafetensors"):
            model = public_load(root, method, TaskType.DENSE_EMBEDDING, tp=2, rank=rank)
            assert model.model_weights_loader.loaded_method == NewLoaderLoadMethod(
                method
            )
            owner = model.py_model

            def reconstruct(local, axis):
                shards = [torch.empty_like(local) for _ in range(2)]
                dist.all_gather(shards, local.contiguous())
                return torch.cat(shards, axis)

            for name in (
                "word_embeddings",
                "token_type_embeddings",
                "position_embeddings",
            ):
                local = getattr(owner.embeddings, name).weight
                torch.testing.assert_close(
                    reconstruct(local, 1), state[f"embeddings.{name}.weight"]
                )
            for i, layer in enumerate(owner.encoder.layer.values()):
                p = f"encoder.layer.{i}."
                for label, part in zip(
                    ("query", "key", "value"), layer.qkv.weight.chunk(3, dim=0)
                ):
                    torch.testing.assert_close(
                        reconstruct(part, 0),
                        state[p + f"attention.self.{label}.weight"],
                    )
                torch.testing.assert_close(
                    reconstruct(layer.attn_output.weight, 1),
                    state[p + "attention.output.dense.weight"],
                )
                torch.testing.assert_close(
                    reconstruct(layer.intermediate.weight, 0),
                    state[p + "intermediate.dense.weight"],
                )
                torch.testing.assert_close(
                    reconstruct(layer.output.weight, 1),
                    state[p + "output.dense.weight"],
                )
    finally:
        dist.destroy_process_group()


class BertServiceTest(unittest.TestCase):
    def test_tp2_real_checkpoint_ownership(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_checkpoint(root)
            torch.multiprocessing.start_processes(
                check_tp_rank,
                args=(d, (root / "gloo").as_uri()),
                nprocs=2,
                start_method="fork",
            )

    def test_missing_attention_projection_fails_loading(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_checkpoint(root)
            path = str(root / "model.safetensors")
            names = load_file(path)
            del names["encoder.layer.1.attention.self.value.weight"]
            save_file(names, path)
            for method in ("scratch", "fastsafetensors"):
                with self.subTest(method=method), self.assertRaises(
                    (ValueError, RuntimeError)
                ):
                    public_load(root, method, TaskType.DENSE_EMBEDDING)


class BertNativeOutputTest(unittest.TestCase):
    @torch.inference_mode()
    def test_registered_dense_and_classifier_outputs(self):
        for classifier in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d)
                ref = make_checkpoint(root, classifier=classifier)
                inputs, lengths = packed_inputs()
                h = []
                logits = []
                offset = 0
                for n in lengths.tolist():
                    part = slice(offset, offset + n)
                    kw = dict(
                        input_ids=inputs.input_ids[part][None],
                        position_ids=inputs.bert_embedding_inputs.combo_position_ids[
                            part
                        ][None],
                        token_type_ids=inputs.bert_embedding_inputs.combo_tokens_type_ids[
                            part
                        ][
                            None
                        ],
                    )
                    base = ref.bert if classifier else ref
                    h.append(base(**kw).last_hidden_state[0])
                    if classifier:
                        logits.append(ref(**kw).logits[0])
                    offset += n
                for method in ("scratch", "fastsafetensors"):
                    with self.subTest(method=method, classifier=classifier):
                        model = public_load(
                            root,
                            method,
                            (
                                TaskType.SEQ_CLASSIFICATION
                                if classifier
                                else TaskType.DENSE_EMBEDDING
                            ),
                        )
                        owner = model.py_model
                        self.assertIsInstance(owner, BertEncoderOwner)
                        self.assertEqual(
                            model.model_weights_loader.loaded_method,
                            NewLoaderLoadMethod(method),
                        )
                        self.assertIs(
                            model.weight.get_global_weight(W.embedding),
                            owner.embeddings.word_embeddings.weight,
                        )
                        self.assertIs(
                            model.weight.get_global_weight(W.pre_decoder_ln_beta),
                            owner.embeddings.LayerNorm.bias,
                        )
                        output = owner(inputs).hidden_states
                        torch.testing.assert_close(
                            output, torch.cat(h), atol=2e-5, rtol=2e-5
                        )
                        pooled = model.custom_module.handler.forward(
                            inputs.input_ids, output, lengths
                        )
                        expected = (
                            torch.stack(logits)
                            if classifier
                            else torch.nn.functional.normalize(
                                torch.stack([row[0] for row in h]), dim=1
                            )
                        )
                        torch.testing.assert_close(
                            pooled, expected.to(pooled.dtype), atol=2e-5, rtol=2e-5
                        )


if __name__ == "__main__":
    unittest.main()
