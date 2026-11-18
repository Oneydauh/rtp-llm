"""Real tiny CPU scratch/FST loads, not GPU/multicard/serving/golden tests."""

import tempfile
import unittest
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.config.quant_config import AWQConfig, GPTQConfig
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.quant_methods.groupwise_checkpoint import (
    GroupwiseCheckpointLayout,
    PackedGroupwiseWeights,
)
from rtp_llm.models_py.quant_methods.test.test_groupwise_checkpoint import (
    checkpoint,
    reconstruct,
)
from rtp_llm.models_py.registry import register_model


@register_model("_groupwise_dense_cpu_owner")
class DenseOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        common = dict(
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            quant_config=load.quant_config,
            params_dtype=torch.float16,
            prefix="proj",
            bias=config.bias,
        )
        if config.kind == "qkv":
            self.proj = QKVParallelLinear(64, 4, 1, 16, **common)
        elif config.kind == "merged":
            self.proj = MergedColumnParallelLinear(
                64, 128, shard_names=["gate_proj", "up_proj"], **common
            )
        else:
            cls = RowParallelLinear if config.kind == "row" else ColumnParallelLinear
            self.proj = cls(64, 64, **common)


@register_model("_groupwise_moe_cpu_owner")
class ExpertOwner(RtpModule):
    def __init__(self, config, load):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=4,
            top_k=1,
            hidden_size=64,
            moe_intermediate_size=64,
            tp_size=load.tp_size,
            tp_rank=load.tp_rank,
            ep_size=load.ep_size,
            ep_rank=load.ep_rank,
            params_dtype=torch.float16,
            model_config=config,
            parallelism_config=SimpleNamespace(
                dp_size=1,
                dp_rank=0,
                world_size=load.tp_size,
                local_world_size=load.tp_size,
            ),
            moe_config=None,
            quant_config=load.quant_config,
            layer_idx=0,
            prefix="experts",
        )


def source_config(method, bits):
    return (AWQConfig if method == "awq" else GPTQConfig)(
        bits=bits, group_size=16, is_quanted=True
    )


def material(method, bits, k, n, offset=0):
    layout = GroupwiseCheckpointLayout(method, bits, 16, k, n)
    values, dense = checkpoint(layout)
    # Distinguish experts/projections without touching packed integer semantics.
    values["scales"] *= offset + 1
    # The legacy signed-weight + FP16 offset ABI has two rounding steps;
    # evaluate that exact reference after changing scales, not dense*factor.
    return values, reconstruct(
        PackedGroupwiseWeights.from_tensors(
            **values, method=method, bits=bits, group_size=16
        )
    )


def distributed_load(rank, world, directory, config, load, selected):
    torch.distributed.init_process_group(
        "gloo",
        init_method="file://" + directory + "/rendezvous",
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=45),
    )
    try:
        load = replace(
            load,
            tp_rank=rank,
            ep_rank=rank if load.ep_size > 1 else 0,
            attn_tp_rank=rank,
            ffn_tp_rank=rank,
            lm_head_tp_rank=rank,
        )
        loader = NewModelLoader(config, load, model_path=directory)
        owner = loader.load()
        assert loader.loaded_method.value == "fastsafetensors"
        if rank == selected:
            torch.save(owner, directory + "/loaded_owner.pt")
    finally:
        torch.distributed.destroy_process_group()


def decode(layer):
    metadata = layer._groupwise_layout
    return reconstruct(
        PackedGroupwiseWeights.from_tensors(
            layer.qweight,
            layer.qzeros,
            layer.scales,
            method=metadata["method"],
            bits=metadata["bits"],
            group_size=metadata["group_size"],
        )
    )


class GroupwiseOwnerTest(unittest.TestCase):
    def load(
        self,
        values,
        config,
        method,
        bits,
        backend,
        tp=1,
        rank=0,
        ep=1,
        ep_rank=0,
        file_format="safetensors",
    ):
        source = source_config(method, bits)
        config.quant_config = source
        config.activation_type = "SiGLU"
        if config.model_type == "_groupwise_moe_cpu_owner":
            config.expert_num = 4
            config.moe_k = 1
        with tempfile.TemporaryDirectory() as directory:
            if file_format == "bin":
                torch.save(values, Path(directory) / "pytorch_model.bin")
            else:
                save_file(values, str(Path(directory) / "model.safetensors"))
            load = NewLoaderConfig(
                device="cpu",
                compute_dtype=torch.float16,
                load_method=backend,
                tp_size=tp,
                tp_rank=rank,
                ep_size=ep,
                ep_rank=ep_rank,
                quant_config=QuantizationConfig(method, source),
            )
            if backend == "fastsafetensors" and tp > 1:
                torch.multiprocessing.start_processes(
                    distributed_load,
                    args=(tp, directory, config, load, rank),
                    nprocs=tp,
                    join=True,
                    start_method="fork",
                )
                return torch.load(directory + "/loaded_owner.pt", weights_only=False)
            loader = NewModelLoader(config, load, model_path=directory)
            owner = loader.load()
            self.assertEqual(loader.loaded_method.value, backend)
            self.assertTrue(all(p.device.type == "cpu" for p in owner.parameters()))
            return owner

    def test_dense_real_scratch_and_fst_all_formats_tp_and_gqa(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                for backend in ("scratch", "fastsafetensors"):
                    for kind in ("column", "row", "merged", "qkv"):
                        tp = 4 if kind == "qkv" else 2
                        parts = (
                            [("q_proj", 64), ("k_proj", 16), ("v_proj", 16)]
                            if kind == "qkv"
                            else (
                                [("gate_proj", 64), ("up_proj", 64)]
                                if kind == "merged"
                                else [("", 64)]
                            )
                        )
                        values, full = {}, []
                        for index, (part, width) in enumerate(parts):
                            packed, dense = material(method, bits, 64, width, index)
                            prefix = "proj." + (part + "." if part else "")
                            values.update(
                                {prefix + key: value for key, value in packed.items()}
                            )
                            values[prefix + "bias"] = (
                                torch.arange(width).half() + index * 100
                            )
                            full.append(dense)
                        for rank in range(tp):
                            with self.subTest(
                                method=method,
                                bits=bits,
                                backend=backend,
                                kind=kind,
                                rank=rank,
                            ):
                                owner = self.load(
                                    values,
                                    SimpleNamespace(
                                        model_type="_groupwise_dense_cpu_owner",
                                        kind=kind,
                                        bias=True,
                                    ),
                                    method,
                                    bits,
                                    backend,
                                    tp,
                                    rank,
                                )
                                expected = []
                                biases = []
                                for index, ((part, width), dense) in enumerate(
                                    zip(parts, full)
                                ):
                                    expected.append(
                                        dense.chunk(tp, 0)[rank]
                                        if kind == "row"
                                        else (
                                            dense
                                            if kind == "qkv" and part != "q_proj"
                                            else dense.chunk(tp, 1)[rank]
                                        )
                                    )
                                    bias = torch.arange(width).half() + index * 100
                                    biases.append(
                                        bias
                                        if kind == "row"
                                        or (kind == "qkv" and part != "q_proj")
                                        else bias.chunk(tp)[rank]
                                    )
                                torch.testing.assert_close(
                                    decode(owner.proj),
                                    torch.cat(expected, 1),
                                    rtol=0,
                                    atol=0,
                                )
                                torch.testing.assert_close(
                                    owner.proj.bias, torch.cat(biases), rtol=0, atol=0
                                )
                                self.assertTrue(owner.proj.checkpoint_load_complete())

    def test_dense_fused_qkv_and_gate_up_sources(self):
        for kind, widths in (("qkv", (64, 16, 16)), ("merged", (64, 64))):
            for method in ("awq", "gptq"):
                packs, dense = zip(
                    *(material(method, 4, 64, n, i) for i, n in enumerate(widths))
                )
                values = {
                    "proj." + key: torch.cat([p[key] for p in packs], dim=1)
                    for key in packs[0]
                }
                for backend in ("scratch", "fastsafetensors"):
                    owner = self.load(
                        values,
                        SimpleNamespace(
                            model_type="_groupwise_dense_cpu_owner",
                            kind=kind,
                            bias=False,
                        ),
                        method,
                        4,
                        backend,
                        2,
                        1,
                    )
                    expected = [
                        v if kind == "qkv" and i else v.chunk(2, 1)[1]
                        for i, v in enumerate(dense)
                    ]
                    torch.testing.assert_close(
                        decode(owner.proj), torch.cat(expected, 1), rtol=0, atol=0
                    )

    def test_grouped_bin_real_scratch(self):
        packed, dense = material("gptq", 4, 64, 64)
        owner = self.load(
            {"proj." + k: v for k, v in packed.items()},
            SimpleNamespace(
                model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
            ),
            "gptq",
            4,
            "scratch",
            file_format="bin",
        )
        torch.testing.assert_close(decode(owner.proj), dense, rtol=0, atol=0)

    def test_moe_real_scratch_and_fst_tp_ep(self):
        for method in ("awq", "gptq"):
            for bits in (4, 8):
                values, full = {}, {}
                for expert in range(4):
                    for index, projection in enumerate(
                        ("gate_proj", "up_proj", "down_proj")
                    ):
                        packed, dense = material(
                            method, bits, 64, 64, expert * 3 + index
                        )
                        values.update(
                            {
                                f"experts.{expert}.{projection}.{key}": value
                                for key, value in packed.items()
                            }
                        )
                        full[expert, projection] = dense
                for backend in ("scratch", "fastsafetensors"):
                    for tp, ep in ((1, 1), (2, 1), (2, 2)):
                        for rank in range(tp):
                            with self.subTest(
                                method=method,
                                bits=bits,
                                backend=backend,
                                tp=tp,
                                ep=ep,
                                rank=rank,
                            ):
                                owner = self.load(
                                    values,
                                    SimpleNamespace(
                                        model_type="_groupwise_moe_cpu_owner"
                                    ),
                                    method,
                                    bits,
                                    backend,
                                    tp,
                                    rank,
                                    ep,
                                    rank if ep > 1 else 0,
                                )
                                layer = owner.experts
                                self.assertTrue(layer.checkpoint_load_complete())
                                for expert in range(4):
                                    for local_id in layer._remap_expert_ids(expert):
                                        for stem in ("w13", "w2"):
                                            decoded = reconstruct(
                                                PackedGroupwiseWeights.from_tensors(
                                                    getattr(layer, stem)[local_id],
                                                    getattr(layer, stem + "_qzeros")[
                                                        local_id
                                                    ],
                                                    getattr(layer, stem + "_scales")[
                                                        local_id
                                                    ],
                                                    method=method,
                                                    bits=bits,
                                                    group_size=16,
                                                )
                                            )
                                            expected = (
                                                full[expert, "down_proj"]
                                                if stem == "w2"
                                                else torch.cat(
                                                    [
                                                        full[expert, "up_proj"],
                                                        full[expert, "gate_proj"],
                                                    ],
                                                    1,
                                                )
                                            )
                                            if ep == 1 and tp > 1:
                                                expected = (
                                                    full[expert, "down_proj"].chunk(
                                                        tp, 0
                                                    )[rank]
                                                    if stem == "w2"
                                                    else torch.cat(
                                                        [
                                                            full[expert, p].chunk(
                                                                tp, 1
                                                            )[rank]
                                                            for p in (
                                                                "up_proj",
                                                                "gate_proj",
                                                            )
                                                        ],
                                                        1,
                                                    )
                                                )
                                            torch.testing.assert_close(
                                                decoded, expected, rtol=0, atol=0
                                            )

    def test_missing_aux_and_activation_order_rejected(self):
        packed, _ = material("gptq", 4, 64, 64)
        cfg = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
        )
        for backend in ("scratch", "fastsafetensors"):
            missing = {
                "proj." + key: value for key, value in packed.items() if key != "qzeros"
            }
            with self.assertRaisesRegex(
                (RuntimeError, ValueError), "missing|not loaded|complete checkpoint"
            ):
                self.load(missing, cfg, "gptq", 4, backend)
            values = {"proj." + key: value for key, value in packed.items()}
            values["proj.g_idx"] = torch.arange(64, dtype=torch.int32).flip(0) // 16
            with self.assertRaisesRegex(ValueError, "activation-order"):
                self.load(values, cfg, "gptq", 4, backend)

    def test_gptq_max_stored_zero_remains_effective_sixteen(self):
        packed, _ = material("gptq", 4, 64, 64)
        packed["qzeros"].fill_(-1)
        for backend in ("scratch", "fastsafetensors"):
            owner = self.load(
                {"proj." + key: value for key, value in packed.items()},
                SimpleNamespace(
                    model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
                ),
                "gptq",
                4,
                backend,
            )
            self.assertTrue(torch.equal(owner.proj.qzeros, packed["qzeros"]))
            reference = (torch.arange(64 * 64).reshape(64, 64) % 16 - 16) * packed[
                "scales"
            ].repeat_interleave(16, 0)
            torch.testing.assert_close(decode(owner.proj), reference, rtol=0, atol=0)

    def test_packed_dtype_and_duplicate_not_silently_converted(self):
        packed, _ = material("awq", 4, 64, 64)
        cfg = SimpleNamespace(
            model_type="_groupwise_dense_cpu_owner", kind="column", bias=False
        )
        for backend in ("scratch", "fastsafetensors"):
            for parameter in ("qweight", "qzeros", "scales"):
                values = {"proj." + key: value for key, value in packed.items()}
                values["proj." + parameter] = packed[parameter].to(torch.int64)
                with self.assertRaisesRegex(TypeError, "int32|floating point"):
                    self.load(values, cfg, "awq", 4, backend)
            owner = self.load(
                {"proj." + key: value for key, value in packed.items()},
                cfg,
                "awq",
                4,
                backend,
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                owner.proj.load_weights({"qweight": packed["qweight"]})
            with self.assertRaisesRegex(RuntimeError, "not a native inference backend"):
                owner.proj(torch.zeros(1, 64, dtype=torch.float16))

    def test_wrong_rank_readspec_rejected_before_copy(self):
        source = source_config("gptq", 4)
        layer = ColumnParallelLinear(
            64,
            64,
            tp_size=2,
            tp_rank=1,
            quant_config=QuantizationConfig("gptq", source),
            params_dtype=torch.float16,
        )
        packed, _ = material("gptq", 4, 64, 64)
        layout = GroupwiseCheckpointLayout("gptq", 4, 16, 64, 64)
        wrong = layout.read_spec("qweight", tuple(packed["qweight"].shape), 1, 2, 0)
        with self.assertRaisesRegex(ValueError, "another projection/rank"):
            layer.load_weights({"qweight": wrong.materialize(packed["qweight"])})


if __name__ == "__main__":
    unittest.main()
