import json
import os
import tempfile
import types
import unittest
from unittest import mock

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.config.quant_config import QuantizationConfig as SourceQuantizationConfig
from rtp_llm.models_py import weight_mapper
from rtp_llm.models_py.eplb_controller import NewLoaderExpertBalancer
from rtp_llm.models_py.expert_placement import (
    ExpertPlacement,
    expert_placement_for_layer,
)
from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewModelLoader,
    _ExpertRangeFilter,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.models_py.quant_methods import QuantizationConfig
from rtp_llm.ops import EplbMode
from rtp_llm.utils.model_weight import W

_MODEL_QUANT_UNSET = object()


def _parallelism(tp_size=1, tp_rank=0, ep_size=1, ep_rank=0, dp_size=1):
    world_size = max(tp_size, ep_size, dp_size)
    return types.SimpleNamespace(
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        dp_size=dp_size,
        dp_rank=0,
        world_size=world_size,
        local_world_size=world_size,
        local_rank=tp_rank,
        get_attn_tp_size=lambda: tp_size,
        get_attn_tp_rank=lambda: tp_rank,
        get_ffn_tp_size=lambda: tp_size,
        get_ffn_tp_rank=lambda: tp_rank,
        prefill_cp_config=types.SimpleNamespace(
            is_enabled=lambda: False,
            is_prefill_enabled=lambda: False,
        ),
        ffn_disaggregate_config=types.SimpleNamespace(enable_ffn_disaggregate=False),
    )


def _moe_config():
    return types.SimpleNamespace(
        ll_num_max_token=1,
        masked_max_token_num=1,
        moe_strategy=0,
        use_mori_ep=False,
        use_deepep_moe=False,
        use_deepep_low_latency=False,
        use_all_gather=True,
    )


def _runtime_model_config(
    num_experts=2,
    quant_config=None,
    hidden_size=8,
    activation_type="SiGLU",
    redundant_expert=0,
    phy2log_path="",
    enable_eplb=False,
):
    return types.SimpleNamespace(
        data_type="fp32",
        quant_config=quant_config,
        exported_device=None,
        expert_num=num_experts,
        moe_k=1,
        moe_topk_group=1,
        hidden_size=hidden_size,
        activation_type=activation_type,
        attn_config=types.SimpleNamespace(head_num=2),
        num_layers=1,
        phy2log_path=phy2log_path,
        eplb_config=types.SimpleNamespace(
            redundant_expert=redundant_expert,
            phy_exp_num=lambda logical: logical + redundant_expert,
            enable_eplb=lambda: enable_eplb,
        ),
    )


def _make_experts(
    num_experts=2,
    top_k=1,
    hidden_size=8,
    moe_intermediate_size=8,
    tp_size=1,
    tp_rank=0,
    ep_size=1,
    ep_rank=0,
    dp_size=1,
    quant_config=None,
    model_quant_config=_MODEL_QUANT_UNSET,
    activation_type="SiGLU",
    redundant_expert=0,
    phy2log_path="",
    enable_eplb=False,
):
    quant_config = quant_config or QuantizationConfig("none")
    if model_quant_config is _MODEL_QUANT_UNSET:
        runtime_method = quant_config.quant_type
        model_quant_config = (
            None
            if runtime_method == "none"
            else types.SimpleNamespace(
                get_runtime_method_key=lambda: runtime_method,
                get_method=lambda: runtime_method,
            )
        )
    return BaseMoEExperts(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden_size,
        moe_intermediate_size=moe_intermediate_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        params_dtype=torch.float32,
        model_config=_runtime_model_config(
            num_experts,
            model_quant_config,
            hidden_size,
            activation_type,
            redundant_expert,
            phy2log_path,
            enable_eplb,
        ),
        parallelism_config=_parallelism(tp_size, tp_rank, ep_size, ep_rank, dp_size),
        moe_config=_moe_config(),
        quant_config=quant_config,
        layer_idx=0,
        prefix="layers.0.mlp.experts",
    )


def _expert_weights(num_experts, hidden_size, intermediate_size):
    result = {}
    for expert_id in range(num_experts):
        offset = float(expert_id * 1000)
        result[f"{expert_id}.gate_proj.weight"] = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .add(offset + 100)
        )
        result[f"{expert_id}.up_proj.weight"] = (
            torch.arange(intermediate_size * hidden_size, dtype=torch.float32)
            .reshape(intermediate_size, hidden_size)
            .add(offset + 200)
        )
        result[f"{expert_id}.down_proj.weight"] = (
            torch.arange(hidden_size * intermediate_size, dtype=torch.float32)
            .reshape(hidden_size, intermediate_size)
            .add(offset + 300)
        )
    return result


class MoEWeightDispatchTest(unittest.TestCase):
    def test_runtime_moe_shuffle_replaces_registered_source_storage(self):
        layer = _make_experts(num_experts=1)
        source_w13 = layer.w13.detach().clone()
        source_w2 = layer.w2.detach().clone()
        source_w13_ptr = layer.w13.data_ptr()
        source_w2_ptr = layer.w2.data_ptr()

        class _ShuffleDevice:
            def shuffle_moe_weight(self, tensor, data_type, name):
                del data_type, name
                return tensor.clone().add_(1)

        layer._model_config.exported_device = _ShuffleDevice()
        weights = layer._build_weights_dict()

        self.assertNotEqual(layer.w13.data_ptr(), source_w13_ptr)
        self.assertNotEqual(layer.w2.data_ptr(), source_w2_ptr)
        self.assertEqual(weights[W.moe_w1].data_ptr(), layer.w13.data_ptr())
        self.assertEqual(weights[W.moe_w2].data_ptr(), layer.w2.data_ptr())
        torch.testing.assert_close(layer.w13, source_w13 + 1)
        torch.testing.assert_close(layer.w2, source_w2 + 1)

    def test_ep_filter_selects_local_experts_before_tensor_loading(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)

        self.assertFalse(
            rank_one.should_load("model.layers.0.mlp.experts.1.gate_proj.weight")
        )
        self.assertTrue(
            rank_one.should_load("model.layers.0.mlp.experts.2.gate_proj.weight")
        )
        self.assertTrue(rank_one.should_load("model.layers.0.self_attn.q_proj.weight"))
        with self.assertRaisesRegex(ValueError, "per-expert checkpoint keys"):
            rank_one.should_load("model.layers.0.mlp.experts.gate_up_proj.weight")
        with self.assertRaisesRegex(ValueError, "outside"):
            rank_one.should_load("model.layers.0.mlp.experts.4.gate_proj.weight")

    def test_ep_filter_routes_redundant_logical_expert_to_every_owner(self):
        model_config = _runtime_model_config(4, redundant_expert=2)
        rank_one = _ExpertRangeFilter(
            4, 2, 1, model_config, _parallelism(ep_size=2, ep_rank=1)
        )

        logical_zero = "model.layers.0.mlp.experts.0.gate_proj.weight"
        logical_one = "model.layers.0.mlp.experts.1.gate_proj.weight"
        logical_two = "model.layers.0.mlp.experts.2.gate_proj.weight"
        self.assertEqual(rank_one.target_ep_ranks(logical_zero), (0, 1))
        self.assertEqual(rank_one.target_ep_ranks(logical_one), (0,))
        self.assertEqual(rank_one.target_ep_ranks(logical_two), (0, 1))
        self.assertTrue(rank_one.should_load(logical_zero))
        self.assertFalse(rank_one.should_load(logical_one))
        self.assertTrue(rank_one.should_load(logical_two))

    def test_identity_placement_ignores_irrelevant_node_partition(self):
        model_config = _runtime_model_config(4)
        parallelism = _parallelism(ep_size=2)
        parallelism.world_size = 4
        parallelism.local_world_size = 1

        placement = expert_placement_for_layer(
            model_config,
            layer_index=0,
            logical_expert_count=4,
            ep_size=2,
            ep_rank=0,
            parallelism_config=parallelism,
        )

        self.assertTrue(placement.is_identity)
        self.assertEqual(placement.local_physical_to_logical, (0, 1))

    def test_safetensors_ep_filter_runs_before_get_tensor(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)
        handle = mock.MagicMock()
        handle.keys.return_value = [
            "model.layers.0.mlp.experts.0.gate_proj.weight",
            "model.layers.0.mlp.experts.2.gate_proj.weight",
            "model.layers.0.self_attn.q_proj.weight",
        ]
        handle.get_tensor.side_effect = lambda name: torch.tensor([len(name)])
        context = mock.MagicMock()
        context.__enter__.return_value = handle

        with mock.patch("safetensors.safe_open", return_value=context):
            loaded = list(
                weight_mapper.get_all_weights(
                    ["model.safetensors"],
                    name_filter=rank_one.should_load,
                )
            )

        self.assertEqual(
            [name for name, _ in loaded],
            [
                "model.layers.0.mlp.experts.2.gate_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
        )
        handle.get_tensor.assert_has_calls(
            [
                mock.call("model.layers.0.mlp.experts.2.gate_proj.weight"),
                mock.call("model.layers.0.self_attn.q_proj.weight"),
            ]
        )
        self.assertEqual(handle.get_tensor.call_count, 2)

    def test_safetensors_ep_expands_stacked_tensor_before_materialization(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)
        stacked_name = "model.layers.0.mlp.experts.gate_up_proj"
        stacked = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        tensor_slice = mock.MagicMock()
        tensor_slice.get_shape.return_value = stacked.shape
        tensor_slice.__getitem__.side_effect = stacked.__getitem__
        handle = mock.MagicMock()
        handle.keys.return_value = [
            stacked_name,
            "model.layers.0.self_attn.q_proj.weight",
        ]
        handle.get_slice.return_value = tensor_slice
        handle.get_tensor.return_value = torch.ones(1)
        context = mock.MagicMock()
        context.__enter__.return_value = handle

        with mock.patch("safetensors.safe_open", return_value=context):
            loaded = list(
                weight_mapper.get_all_weights(
                    ["model.safetensors"],
                    name_filter=NewModelLoader._checkpoint_name_filter(
                        NewModelLoader._model_checkpoint_name_filter(RtpModule()),
                        rank_one,
                    ),
                    safetensor_slice_expander=rank_one,
                )
            )

        self.assertEqual(
            [name for name, _ in loaded],
            [
                "model.layers.0.mlp.experts.2.gate_up_proj.weight",
                "model.layers.0.mlp.experts.3.gate_up_proj.weight",
                "model.layers.0.self_attn.q_proj.weight",
            ],
        )
        torch.testing.assert_close(loaded[0][1], stacked[2])
        torch.testing.assert_close(loaded[1][1], stacked[3])
        handle.get_tensor.assert_called_once_with(
            "model.layers.0.self_attn.q_proj.weight"
        )

    def test_safetensors_ep_preserves_stacked_auxiliary_suffix(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)

        expanded = rank_one.expand_safetensor(
            "model.layers.0.mlp.experts.down_proj.weight_scale_inv",
            (4, 2, 3),
        )

        self.assertEqual(
            expanded,
            [
                (
                    "model.layers.0.mlp.experts.2.down_proj.weight_scale_inv",
                    2,
                ),
                (
                    "model.layers.0.mlp.experts.3.down_proj.weight_scale_inv",
                    3,
                ),
            ],
        )

    def test_safetensors_ep_rejects_wrong_stacked_expert_count(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)

        with self.assertRaisesRegex(ValueError, "must contain 4 experts"):
            rank_one.expand_safetensor(
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                (3, 2, 3),
            )

    def test_model_filter_runs_before_stacked_expert_expansion(self):
        rank_one = _ExpertRangeFilter(4, 2, 1)
        handle = mock.MagicMock()
        handle.keys.return_value = ["mtp.layers.0.mlp.experts.gate_up_proj"]
        context = mock.MagicMock()
        context.__enter__.return_value = handle

        with mock.patch("safetensors.safe_open", return_value=context):
            loaded = list(
                weight_mapper.get_all_weights(
                    ["model.safetensors"],
                    name_filter=lambda name: not name.startswith("mtp."),
                    safetensor_slice_expander=rank_one,
                )
            )

        self.assertEqual(loaded, [])
        handle.get_slice.assert_not_called()
        handle.get_tensor.assert_not_called()

    def test_rms_res_norm_preserves_decoder_residual_contract(self):
        layer = RMSResNorm(2, eps=1e-6, params_dtype=torch.float32)
        hidden = torch.tensor([[1.0, 2.0]])
        residual = torch.tensor([[3.0, 4.0]])

        normalized, residual_out = layer(hidden, residual)

        expected_residual = hidden + residual
        variance = expected_residual.float().pow(2).mean(-1, keepdim=True)
        expected = expected_residual * torch.rsqrt(variance + 1e-6)
        torch.testing.assert_close(residual_out, expected_residual)
        torch.testing.assert_close(normalized, expected)

    def test_rms_res_norm_rejects_mismatched_dtype(self):
        layer = RMSResNorm(2, params_dtype=torch.float32)
        with self.assertRaisesRegex(TypeError, "share a dtype"):
            layer(torch.ones(1, 2), torch.ones(1, 2, dtype=torch.float64))

    def test_rms_res_norm_rejects_non_matrix_and_weight_dtype_mismatch(self):
        layer = RMSResNorm(2, params_dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "expected \\[tokens, 2\\]"):
            layer(torch.ones(1, 1, 2), torch.ones(1, 1, 2))

        layer = RMSResNorm(2, params_dtype=torch.float64)
        with self.assertRaisesRegex(TypeError, "weight and inputs"):
            layer(torch.ones(1, 2), torch.ones(1, 2))

    def test_ep_rejects_pytorch_checkpoint_before_deserialization(self):
        with tempfile.TemporaryDirectory() as model_path:
            checkpoint = os.path.join(model_path, "model.pt")
            with open(checkpoint, "wb"):
                pass
            loader = NewModelLoader(
                types.SimpleNamespace(
                    model_type="qwen_3_moe",
                    model_path=model_path,
                    expert_num=2,
                ),
                NewLoaderConfig(
                    ep_size=2,
                    ep_rank=0,
                    device="cpu",
                ),
            )
            with mock.patch.object(
                weight_mapper, "_load_pytorch", side_effect=AssertionError
            ):
                with self.assertRaisesRegex(ValueError, "EP streaming requires"):
                    loader.load()

    def test_per_expert_weights_pack_up_before_gate(self):
        layer = _make_experts(num_experts=2)
        weights = _expert_weights(2, 8, 8)

        for name, tensor in weights.items():
            layer.load_weights({name: tensor})
        layer._check_load_complete()

        for expert_id in range(2):
            torch.testing.assert_close(
                layer.w13[expert_id, :8],
                weights[f"{expert_id}.up_proj.weight"],
            )
            torch.testing.assert_close(
                layer.w13[expert_id, 8:],
                weights[f"{expert_id}.gate_proj.weight"],
            )
            torch.testing.assert_close(
                layer.w2[expert_id],
                weights[f"{expert_id}.down_proj.weight"],
            )

    def test_tp_slices_gate_up_rows_and_down_columns(self):
        weights = _expert_weights(1, 8, 8)
        for rank in range(2):
            layer = _make_experts(num_experts=1, tp_size=2, tp_rank=rank)
            layer.load_weights(weights)
            start = rank * 4
            torch.testing.assert_close(
                layer.w13[0, :4], weights["0.up_proj.weight"][start : start + 4]
            )
            torch.testing.assert_close(
                layer.w13[0, 4:], weights["0.gate_proj.weight"][start : start + 4]
            )
            torch.testing.assert_close(
                layer.w2[0], weights["0.down_proj.weight"][:, start : start + 4]
            )

    def test_ep_loads_only_local_experts(self):
        weights = _expert_weights(4, 8, 8)
        layer = _make_experts(num_experts=4, ep_size=2, ep_rank=1)
        layer.load_weights(weights)
        layer._check_load_complete()

        torch.testing.assert_close(layer.w13[0, :8], weights["2.up_proj.weight"])
        torch.testing.assert_close(layer.w13[1, :8], weights["3.up_proj.weight"])

    def test_static_redundant_expert_duplicates_weights_into_physical_slots(self):
        weights = _expert_weights(4, 8, 8)
        layer = _make_experts(num_experts=4, ep_size=2, ep_rank=1, redundant_expert=2)
        layer.load_weights(weights)
        layer._check_load_complete()

        self.assertEqual(layer.physical_num_experts, 6)
        self.assertEqual(layer.num_local_experts, 3)
        self.assertEqual(layer._expert_placement.local_physical_to_logical, (2, 3, 0))
        for local_id, logical_id in enumerate((2, 3, 0)):
            torch.testing.assert_close(
                layer.w13[local_id, :8], weights[f"{logical_id}.up_proj.weight"]
            )
            torch.testing.assert_close(
                layer.w13[local_id, 8:], weights[f"{logical_id}.gate_proj.weight"]
            )
            torch.testing.assert_close(
                layer.w2[local_id], weights[f"{logical_id}.down_proj.weight"]
            )

    def test_forward_maps_logical_topk_ids_to_legacy_initial_slots(self):
        layer = _make_experts(num_experts=4, redundant_expert=2)
        layer.fused_moe = mock.Mock(side_effect=lambda **kwargs: kwargs["topk_ids"])
        logical_ids = torch.tensor([[0], [0], [1], [1]], dtype=torch.int32)

        physical_ids = layer(torch.zeros(4, 8), torch.ones(4, 1), logical_ids)

        self.assertEqual(physical_ids.dtype, logical_ids.dtype)
        torch.testing.assert_close(
            physical_ids, torch.tensor([[0], [0], [1], [1]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            layer._map_logical_to_physical(
                torch.tensor([[-1], [4]], dtype=torch.int32)
            ),
            torch.tensor([[-1], [4]], dtype=torch.int32),
        )

    def test_forward_static_initial_slot_selection_is_rank_consistent(self):
        logical_ids = torch.tensor([[0], [0], [2], [2]], dtype=torch.int32)
        expected = torch.tensor([[0], [0], [3], [3]], dtype=torch.int32)
        for rank in (0, 1):
            layer = _make_experts(
                num_experts=4, ep_size=2, ep_rank=rank, redundant_expert=2
            )
            layer.fused_moe = mock.Mock(side_effect=lambda **kwargs: kwargs["topk_ids"])
            physical_ids = layer(torch.zeros(4, 8), torch.ones(4, 1), logical_ids)
            torch.testing.assert_close(physical_ids, expected)

    def test_forward_dynamic_plan_preserves_logical_aggregation_metadata(self):
        layer = _make_experts(
            num_experts=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        )
        placement = ExpertPlacement(
            logical_expert_count=4,
            ep_size=2,
            ep_rank=0,
            physical_to_logical=(0, 1, 0, 2, 3, 3),
        )
        counts, log2phy = layer._routing_tensors(placement)
        layer._expert_placement = placement
        layer._eplb_logic_expert_cnt.copy_(counts)
        layer._eplb_log2phy.copy_(log2phy)
        layer._eplb_plan_installed = True
        layer.fused_moe = mock.Mock(
            side_effect=lambda **kwargs: kwargs["extra_finalize_args"]
        )
        logical_ids = torch.tensor([[0], [3]], dtype=torch.int32)

        aggregation = layer(
            torch.zeros(2, 8), torch.ones(2, 1), logical_ids
        )

        self.assertIs(aggregation["eplb_logical_topk_ids"], logical_ids)
        self.assertEqual(aggregation["eplb_logical_experts_per_rank"], 2)

    def test_invalid_ep_and_expert_count_fail_fast(self):
        with self.assertRaisesRegex(
            ValueError, "num_experts must be a positive integer"
        ):
            _make_experts(num_experts=0)
        with self.assertRaisesRegex(ValueError, "divisible by ep_size"):
            _make_experts(num_experts=3, ep_size=2)
        with self.assertRaisesRegex(ValueError, "ep_rank"):
            _make_experts(num_experts=4, ep_size=2, ep_rank=2)
        for kwargs in (
            {"num_experts": True},
            {"tp_size": True},
            {"tp_size": 2, "tp_rank": True},
            {"ep_size": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(
                ValueError, "positive integer|rank"
            ):
                _make_experts(**kwargs)

    def test_malformed_unknown_and_out_of_range_names_fail(self):
        layer = _make_experts(num_experts=2)
        cases = {
            "bad": torch.zeros(8, 8),
            "0.unknown.weight": torch.zeros(8, 8),
            "2.gate_proj.weight": torch.zeros(8, 8),
            "0.gate_proj.weight.extra": torch.zeros(8, 8),
        }
        for name, tensor in cases.items():
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                _make_experts(num_experts=2).load_weights({name: tensor})

    def test_duplicate_and_missing_weights_fail(self):
        layer = _make_experts(num_experts=1)
        gate = torch.zeros(8, 8)
        layer.load_weights({"0.gate_proj.weight": gate})
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            layer.load_weights({"0.gate_proj.weight": gate})
        with self.assertRaisesRegex(RuntimeError, "missing"):
            layer._check_load_complete()

    def test_fused_and_separate_projection_aliases_cannot_overwrite(self):
        layer = _make_experts(num_experts=1, hidden_size=12, moe_intermediate_size=4)
        gate = torch.ones(4, 12)
        layer.load_weights({"0.gate_proj.weight": gate})
        before = layer.w13[0, 4:].clone()

        with self.assertRaisesRegex(RuntimeError, "Duplicate MoE projection"):
            layer.load_weights({"0.gate_up_proj.weight": torch.zeros(8, 12)})
        torch.testing.assert_close(layer.w13[0, 4:], before)

    def test_stacked_weight_suffix_supports_both_layouts(self):
        for transpose_layout in (False, True):
            layer = _make_experts(
                num_experts=2, hidden_size=12, moe_intermediate_size=4
            )
            gate = torch.arange(2 * 4 * 12, dtype=torch.float32).reshape(2, 4, 12)
            up = gate + 1000
            down = torch.arange(2 * 12 * 4, dtype=torch.float32).reshape(2, 12, 4)
            down = down + 2000
            gate_up = torch.cat((gate, up), dim=1)
            stacked_gate_up = (
                gate_up.transpose(1, 2).contiguous() if transpose_layout else gate_up
            )
            stacked_down = (
                down.transpose(1, 2).contiguous() if transpose_layout else down
            )
            layer.load_weights(
                {
                    "gate_up_proj.weight": stacked_gate_up,
                    "down_proj.weight": stacked_down,
                }
            )
            layer._check_load_complete()
            torch.testing.assert_close(layer.w13[:, :4], up)
            torch.testing.assert_close(layer.w13[:, 4:], gate)
            torch.testing.assert_close(layer.w2, down)

    def test_ambiguous_stacked_layout_fails_instead_of_guessing(self):
        layer = _make_experts(num_experts=1, hidden_size=8, moe_intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            layer.load_weights({"gate_up_proj.weight": torch.zeros(1, 8, 8)})

    def test_ambiguous_per_expert_fused_layout_fails(self):
        layer = _make_experts(num_experts=1, hidden_size=8, moe_intermediate_size=4)
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            layer.load_weights({"0.gate_up_proj.weight": torch.zeros(8, 8)})

    def test_forward_requires_completed_post_load_processing(self):
        layer = _make_experts(num_experts=1)
        with self.assertRaisesRegex(RuntimeError, "post-load processing"):
            layer(
                torch.zeros(1, 8),
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
            )

    def test_forward_rejects_non_matrix_hidden_states(self):
        layer = _make_experts(num_experts=1)
        layer.fused_moe = mock.MagicMock()
        with self.assertRaisesRegex(ValueError, "\[tokens, 8\]"):
            layer(
                torch.zeros(1, 1, 8),
                torch.ones(1, 1),
                torch.zeros(1, 1, dtype=torch.int32),
            )

    def test_ep_rejects_stacked_all_expert_checkpoint(self):
        layer = _make_experts(num_experts=2, ep_size=2, ep_rank=0)
        with self.assertRaisesRegex(ValueError, "per-expert checkpoint keys"):
            layer.load_weights({"gate_up_proj.weight": torch.zeros(2, 16, 8)})

    def test_redundant_experts_reject_stacked_all_expert_checkpoint(self):
        layer = _make_experts(num_experts=2, redundant_expert=2)
        with self.assertRaisesRegex(ValueError, "redundant physical"):
            layer.load_weights({"gate_up_proj.weight": torch.zeros(2, 16, 8)})


class MoEQuantizedDispatchTest(unittest.TestCase):
    def test_prequantized_fp8_rejects_unquantized_weight_dtype(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        with self.assertRaisesRegex(TypeError, "requires FP8 storage"):
            layer.load_weights({"0.gate_proj.weight": torch.zeros(8, 8)})

    def test_fp8_scale_dtype_is_validated_during_dispatch(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        with self.assertRaisesRegex(TypeError, "must be floating point"):
            layer.load_weights(
                {"0.gate_proj.weight_scale": torch.tensor(1, dtype=torch.int32)}
            )

    def test_fp8_scale_value_validation_is_batched_after_dispatch(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        with mock.patch.object(
            torch,
            "isfinite",
            side_effect=AssertionError("dispatch synchronized scale validation"),
        ):
            layer.load_weights({"0.gate_proj.weight_scale": torch.tensor(1.0)})

        for scale in (
            torch.tensor(0.0),
            torch.tensor(-1.0),
            torch.tensor(float("nan")),
            torch.tensor(float("inf")),
        ):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "finite and positive"
            ):
                layer.quant_method._validate_positive_finite_scales(
                    (torch.ones(2), scale), "per-tensor"
                )

    def test_fp8_scale_batch_validation_accepts_positive_finite_values(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8_per_channel"),
        )
        layer.quant_method._validate_positive_finite_scales(
            (torch.ones(2, 3), torch.full((4,), 0.5)), "per-channel"
        )

    def test_fp8_per_channel_postprocess_rejects_invalid_loaded_scale(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8_per_channel"),
        )
        layer._up_ch_scales.fill_(1)
        layer._gate_ch_scales.fill_(1)
        layer._down_ch_scales.fill_(1)
        layer._gate_ch_scales[0, 0] = float("nan")

        with mock.patch.object(layer.quant_method, "_fuse_per_channel") as fuse:
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                layer.quant_method.process_weights_after_loading(layer)
        fuse.assert_not_called()

    def test_single_channel_fp8_scale_keeps_vector_shape(self):
        layer = _make_experts(
            num_experts=1,
            hidden_size=1,
            moe_intermediate_size=1,
            quant_config=QuantizationConfig("fp8_per_channel"),
        )
        scale = torch.ones(1, 1)

        layer.load_weights(
            {
                "0.gate_proj.weight_scale": scale,
                "0.up_proj.weight_scale": scale,
                "0.down_proj.weight_scale": scale,
            }
        )

        self.assertEqual(tuple(layer._gate_ch_scales.shape), (1, 1))
        self.assertEqual(tuple(layer._up_ch_scales.shape), (1, 1))
        self.assertEqual(tuple(layer._down_ch_scales.shape), (1, 1))

    def test_custom_block_conversion_still_applies_ue8m0(self):
        layer = torch.nn.Module()
        layer._FP8_BLOCK_SIZE = 128
        layer._fp8_moe_weight_block_size = [64, 64]
        layer.w13 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_scale = torch.ones(1, 2, 2)
        layer.w2_scale = torch.ones(1, 2, 2)
        layer.num_local_experts = 1
        converted = []

        def convert(weight, scale):
            converted.append((weight, scale))
            return weight, scale.to(torch.uint8)

        from rtp_llm.models_py.quant_methods.fp8_moe import Fp8MoEMethod

        method = Fp8MoEMethod.__new__(Fp8MoEMethod)
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8_moe._runtime_fp8_dtype",
            return_value=torch.float8_e4m3fn,
        ), mock.patch.multiple(
            "rtp_llm.models_py.quant_methods.fp8",
            per_block_quant_like_legacy=mock.DEFAULT,
            is_deep_gemm_e8m0_used=mock.DEFAULT,
            _resolve_requant_weight_ue8m0=mock.DEFAULT,
        ) as dependencies:
            dependencies["per_block_quant_like_legacy"].side_effect = (
                lambda weight, block: (
                    weight.to(torch.float8_e4m3fn),
                    torch.ones(1, 1),
                )
            )
            dependencies["is_deep_gemm_e8m0_used"].return_value = True
            dependencies["_resolve_requant_weight_ue8m0"].return_value = convert
            method._requant_per_block_if_needed(layer)

        self.assertEqual(len(converted), 2)
        self.assertEqual(layer.w13_scale.dtype, torch.uint8)
        self.assertEqual(layer.w2_scale.dtype, torch.uint8)

    @unittest.skipUnless(hasattr(torch, "float8_e4m3fnuz"), "FNUZ dtype is unavailable")
    def test_custom_block_conversion_preserves_runtime_fp8_dtype(self):
        layer = torch.nn.Module()
        layer._FP8_BLOCK_SIZE = 128
        layer._fp8_moe_weight_block_size = [64, 64]
        layer.w13 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2 = torch.nn.Parameter(
            torch.zeros(1, 128, 128, dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_scale = torch.ones(1, 2, 2)
        layer.w2_scale = torch.ones(1, 2, 2)
        layer.num_local_experts = 1

        from rtp_llm.models_py.quant_methods.fp8_moe import Fp8MoEMethod

        method = Fp8MoEMethod.__new__(Fp8MoEMethod)
        runtime_dtype = torch.float8_e4m3fnuz
        with mock.patch(
            "rtp_llm.models_py.quant_methods.fp8_moe._runtime_fp8_dtype",
            return_value=runtime_dtype,
        ), mock.patch.multiple(
            "rtp_llm.models_py.quant_methods.fp8",
            per_block_quant_like_legacy=mock.DEFAULT,
            is_deep_gemm_e8m0_used=mock.DEFAULT,
            _resolve_requant_weight_ue8m0=mock.DEFAULT,
        ) as dependencies:
            dependencies["per_block_quant_like_legacy"].side_effect = (
                lambda weight, block: (
                    weight.to(runtime_dtype),
                    torch.ones(1, 1),
                )
            )
            dependencies["is_deep_gemm_e8m0_used"].return_value = False
            method._requant_per_block_if_needed(layer)

        self.assertEqual(layer.w13.dtype, runtime_dtype)
        self.assertEqual(layer.w2.dtype, runtime_dtype)

    def test_w4a8_rejects_unsupported_cuda_capability_before_import(self):
        from rtp_llm.models_py.quant_methods.w4a8_moe import W4A8Int4MoEMethod

        method = W4A8Int4MoEMethod(QuantizationConfig("W4A8_INT4_PER_CHANNEL"))
        with mock.patch.object(
            torch.cuda, "is_available", return_value=True
        ), mock.patch.object(
            torch.cuda, "current_device", return_value=0
        ), mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(8, 0)
        ), mock.patch.object(
            torch.version, "hip", None
        ):
            with self.assertRaisesRegex(RuntimeError, "capability 8.9"):
                method.validate_runtime_device(torch.device("cuda:0"))

    def test_online_and_unquantized_moe_reject_integer_weights(self):
        for quant_type in ("none", "fp8_online", "W4A8_INT4_PER_CHANNEL"):
            with self.subTest(quant_type=quant_type), self.assertRaisesRegex(
                TypeError, "floating"
            ):
                size = 128 if quant_type == "W4A8_INT4_PER_CHANNEL" else 8
                layer = _make_experts(
                    num_experts=1,
                    hidden_size=size,
                    moe_intermediate_size=size,
                    quant_config=QuantizationConfig(quant_type),
                )
                layer.load_weights(
                    {"0.gate_proj.weight": torch.zeros(size, size, dtype=torch.int8)}
                )

    def test_w4a8_repack_rejects_invalid_scales_before_kernel_import(self):
        from rtp_llm.models_py.quant_methods.w4a8_utils import (
            repack_compressed_int4_to_cutlass,
        )

        packed = torch.zeros(1, 64, dtype=torch.int8)
        for scale in (
            torch.tensor([[0.0]]),
            torch.tensor([[float("inf")]]),
            torch.tensor([[1000.0]]),
        ):
            with self.subTest(scale=scale), self.assertRaisesRegex(
                ValueError, "finite and positive|representable"
            ):
                repack_compressed_int4_to_cutlass(packed, scale, 128)

    def test_w4a8_repack_allows_positive_scale_to_underflow_to_fp8_zero(self):
        from rtp_llm.models_py.quant_methods.w4a8_utils import (
            repack_compressed_int4_to_cutlass,
        )

        packed = torch.zeros(1, 64, dtype=torch.int8)
        scale = torch.tensor([[1.0e-6]])
        fake_kernel = types.SimpleNamespace(
            pack_scale_fp8=lambda value: value,
            reorder_tensor=lambda value: value,
            unified_encode_int4b=lambda value: value,
        )
        with mock.patch.dict(
            "sys.modules", {"rtp_kernel.w4a8_group_gemm": fake_kernel}
        ):
            _, repacked_scale = repack_compressed_int4_to_cutlass(packed, scale, 128)

        self.assertEqual(repacked_scale.dtype, torch.float8_e4m3fn)
        self.assertEqual(repacked_scale.item(), 0.0)

    def test_invalid_fp8_block_size_types_fail_fast(self):
        for block_size in (128, [True, 128], [64.5, 128], [128]):
            with self.subTest(block_size=block_size), self.assertRaises(
                (TypeError, ValueError)
            ):
                quant = QuantizationConfig(
                    "fp8_block",
                    source_config=types.SimpleNamespace(weight_block_size=block_size),
                )
                _make_experts(num_experts=1, quant_config=quant)

    def test_runtime_and_model_quantization_must_match(self):
        source_quant = types.SimpleNamespace(get_runtime_method_key=lambda: "fp8_block")
        with self.assertRaisesRegex(ValueError, "quantization mismatch"):
            _make_experts(
                num_experts=1,
                quant_config=QuantizationConfig("fp8"),
                model_quant_config=source_quant,
            )
        with self.assertRaisesRegex(ValueError, "without model_config.quant_config"):
            _make_experts(
                num_experts=1,
                quant_config=QuantizationConfig("fp8"),
                model_quant_config=None,
            )

    def test_missing_fp8_scale_fails(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        fp8 = torch.zeros(8, 8, dtype=torch.float8_e4m3fn)
        layer.load_weights(
            {
                "0.gate_proj.weight": fp8,
                "0.up_proj.weight": fp8,
                "0.down_proj.weight": fp8,
            }
        )
        with self.assertRaisesRegex(RuntimeError, "auxiliary tensors"):
            layer._check_load_complete()

    def test_fused_and_separate_scale_aliases_cannot_overwrite(self):
        layer = _make_experts(
            num_experts=1,
            hidden_size=12,
            moe_intermediate_size=4,
            quant_config=QuantizationConfig("fp8"),
        )
        layer.load_weights({"0.gate_proj.weight_scale": torch.tensor(1.0)})
        with self.assertRaisesRegex(RuntimeError, "Duplicate MoE auxiliary"):
            layer.load_weights(
                {"0.gate_up_proj.weight_scale": torch.tensor([2.0, 3.0])}
            )

    def test_fused_block_scale_uses_global_grid_before_tp_slice(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=16,
            tp_size=2,
            tp_rank=1,
            quant_config=quant,
        )
        fused = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8).add(1)
        layer.load_weights({"gate_up_proj.weight_scale_inv": fused})

        gate_global = fused[0, :, :4].t()
        up_global = fused[0, :, 4:].t()
        torch.testing.assert_close(layer.w13_scale[0, :2], up_global[2:4])
        torch.testing.assert_close(layer.w13_scale[0, 2:], gate_global[2:4])

    def test_single_tp_allows_partial_final_fp8_block(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=10,
            quant_config=quant,
        )
        layer.load_weights(
            {
                "0.gate_proj.weight_scale_inv": torch.ones(3, 2),
                "0.up_proj.weight_scale_inv": torch.ones(3, 2),
                "0.down_proj.weight_scale_inv": torch.ones(2, 3),
            }
        )
        self.assertEqual(len(layer._loaded_aux_keys), 3)

    def test_non_aligned_fp8_block_tp_is_rejected(self):
        quant = QuantizationConfig(
            "fp8_block",
            source_config=types.SimpleNamespace(weight_block_size=[4, 4]),
        )
        with self.assertRaisesRegex(ValueError, "block-aligned"):
            _make_experts(
                num_experts=1,
                hidden_size=8,
                moe_intermediate_size=10,
                tp_size=2,
                quant_config=quant,
            )

    def test_stacked_w4a8_weight_packed_is_not_treated_as_scale(self):
        quant = QuantizationConfig(
            "W4A8_INT4_PER_CHANNEL_COMPRESSED",
            source_config=types.SimpleNamespace(group_size=lambda: 4),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=8,
            quant_config=quant,
        )
        gate_up = torch.arange(16 * 4, dtype=torch.int8).reshape(1, 16, 4)
        down = torch.arange(8 * 4, dtype=torch.int8).reshape(1, 8, 4)
        gate_up_scale = torch.ones(1, 16, 2, dtype=torch.bfloat16)
        down_scale = torch.ones(1, 8, 2, dtype=torch.bfloat16)
        layer.load_weights(
            {
                "gate_up_proj.weight_packed": gate_up,
                "down_proj.weight_packed": down,
                "gate_up_proj.weight_scale": gate_up_scale,
                "down_proj.weight_scale": down_scale,
            }
        )
        layer._check_load_complete()
        self.assertTrue(layer.requires_staged_device_postprocess())

        torch.testing.assert_close(layer._w4a8_gate_packed[0], gate_up[0, :8])
        torch.testing.assert_close(layer._w4a8_up_packed[0], gate_up[0, 8:])
        torch.testing.assert_close(layer._w4a8_down_packed[0], down[0])

    def test_compressed_w4a8_fuses_up_gate_before_cutlass_repack(self):
        quant = QuantizationConfig(
            "W4A8_INT4_PER_CHANNEL_COMPRESSED",
            source_config=types.SimpleNamespace(group_size=lambda: 4),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=8,
            quant_config=quant,
        )
        gate = torch.full((8, 4), 11, dtype=torch.int8)
        up = torch.full((8, 4), 22, dtype=torch.int8)
        down = torch.full((8, 4), 33, dtype=torch.int8)
        gate_scale = torch.full((8, 2), 1.0, dtype=torch.bfloat16)
        up_scale = torch.full((8, 2), 2.0, dtype=torch.bfloat16)
        down_scale = torch.full((8, 2), 3.0, dtype=torch.bfloat16)
        layer.load_weights(
            {
                "0.gate_proj.weight_packed": gate,
                "0.up_proj.weight_packed": up,
                "0.down_proj.weight_packed": down,
                "0.gate_proj.weight_scale": gate_scale,
                "0.up_proj.weight_scale": up_scale,
                "0.down_proj.weight_scale": down_scale,
            }
        )

        calls = []

        def fake_repack(weight, scale, group_size):
            calls.append((weight.clone(), scale.clone(), group_size))
            packed_scale = (
                scale.t()
                .contiguous()
                .unsqueeze(-1)
                .expand(-1, -1, 8)
                .to(torch.float8_e4m3fn)
                .contiguous()
            )
            return weight.clone(), packed_scale

        with mock.patch(
            "rtp_llm.models_py.quant_methods.w4a8_moe."
            "repack_compressed_int4_to_cutlass",
            side_effect=fake_repack,
        ):
            layer.quant_method.process_weights_after_loading(layer)

        self.assertEqual(len(calls), 2)
        fused_weight, fused_scale, group_size = calls[0]
        self.assertEqual(group_size, 4)
        torch.testing.assert_close(fused_weight[:8], up)
        torch.testing.assert_close(fused_weight[8:], gate)
        torch.testing.assert_close(fused_scale[:8], up_scale)
        torch.testing.assert_close(fused_scale[8:], gate_scale)
        torch.testing.assert_close(layer.w13[0], torch.cat((up, gate), dim=0))

    def test_w4a8_weight_shape_metadata_is_validated_and_consumed(self):
        quant = QuantizationConfig(
            "W4A8_INT4_PER_CHANNEL_COMPRESSED",
            source_config=types.SimpleNamespace(group_size=lambda: 4),
        )
        layer = _make_experts(
            num_experts=1,
            hidden_size=8,
            moe_intermediate_size=8,
            quant_config=quant,
        )
        layer.load_weights(
            {"0.down_proj.weight_shape": torch.tensor([8, 8], dtype=torch.int64)}
        )
        with self.assertRaisesRegex(ValueError, "describes 56 elements"):
            layer.load_weights(
                {"0.gate_proj.weight_shape": torch.tensor([7, 8], dtype=torch.int64)}
            )

    def test_ignored_moe_layer_uses_unquantized_method_and_runtime_config(self):
        source_quant = types.SimpleNamespace(get_method=lambda: "FP8")
        quant = QuantizationConfig(
            "fp8",
            ignored_layers=["model.layers.{i}.mlp"],
        )
        layer = _make_experts(
            num_experts=1,
            quant_config=quant,
            model_quant_config=source_quant,
        )

        self.assertEqual(layer._quant_family, "none")
        self.assertEqual(layer._required_aux_param_names(), ())
        self.assertIsNone(layer._effective_model_quant_config)
        self.assertEqual(layer.w13.dtype, torch.float32)

    def test_partial_projection_ignore_is_rejected(self):
        quant = QuantizationConfig(
            "fp8",
            ignored_layers=["layers.0.mlp.experts.gate_proj"],
        )
        with self.assertRaisesRegex(ValueError, "partially match fused MoE"):
            _make_experts(num_experts=1, quant_config=quant)

    def test_all_projection_ignores_disable_quantization(self):
        patterns = [
            f"layers.0.mlp.experts.{projection}"
            for projection in BaseMoEExperts.PROJ_NAMES
        ]
        quant = QuantizationConfig("fp8", ignored_layers=patterns)
        layer = _make_experts(num_experts=1, quant_config=quant)
        self.assertEqual(layer._quant_family, "none")

    def test_w4a8_is_moe_only_for_linear_dispatch(self):
        quant = QuantizationConfig("W4A8_INT4_PER_CHANNEL")
        method = quant.get_quant_method(
            types.SimpleNamespace(shard_names=()), "layers.0.self_attn.o_proj"
        )
        self.assertEqual(type(method).__name__, "UnquantizedLinearMethod")

    def test_cuda_graph_flag_is_forwarded_to_moe_adapter(self):
        quant = QuantizationConfig(
            "none",
            hw_kernel_config=types.SimpleNamespace(enable_cuda_graph=True),
        )
        layer = _make_experts(num_experts=1, quant_config=quant)
        self.assertTrue(layer._enable_cuda_graph())

    def test_unsupported_moe_activation_fails_at_construction(self):
        with self.assertRaisesRegex(ValueError, "supports activation_type='SiGLU'"):
            _make_experts(num_experts=1, activation_type="GELU")

    def test_static_activation_fp8_scale_fails_with_explicit_error(self):
        layer = _make_experts(
            num_experts=1,
            quant_config=QuantizationConfig("fp8"),
        )
        with self.assertRaisesRegex(RuntimeError, "Static-activation FP8 MoE"):
            layer._dispatch_scale(
                0,
                "gate_proj",
                "input_scale",
                torch.ones(1, dtype=torch.float32),
            )


class MoeRuntimeConfigTest(unittest.TestCase):
    def test_eplb_router_preflight_never_writes_live_weights(self):
        layer = _make_experts(num_experts=2, redundant_expert=2, enable_eplb=True)
        layer.w13.fill_(7)
        layer.w2.fill_(8)
        layer._runtime_weights_dict = {W.moe_w1: layer.w13.data, W.moe_w2: layer.w2.data}
        layer.fused_moe = types.SimpleNamespace(
            fused_experts=types.SimpleNamespace(use_deterministic_eplb_gather=False),
            router=types.SimpleNamespace(),
        )
        placement = ExpertPlacement(2, 1, 0, (0, 1, 0, 1))
        counts, inverse = layer._routing_tensors(placement)
        with self.assertRaisesRegex(RuntimeError, "aggregation-capable"):
            layer.apply_eplb_plan(
                torch.tensor(placement.physical_to_logical, dtype=torch.int32), counts, inverse,
                {W.moe_w1: torch.zeros_like(layer.w13), W.moe_w2: torch.zeros_like(layer.w2)},
            )
        torch.testing.assert_close(layer.w13, torch.full_like(layer.w13, 7))
        torch.testing.assert_close(layer.w2, torch.full_like(layer.w2, 8))
        self.assertFalse(layer._eplb_plan_installed)

    def test_eplb_peer_install_failure_restores_tensor_routing_and_executor_flags(self):
        layer = _make_experts(num_experts=2, redundant_expert=2, enable_eplb=True)
        layer.w13.fill_(7)
        layer.w2.fill_(8)
        layer._runtime_weights_dict = {W.moe_w1: layer.w13.data, W.moe_w2: layer.w2.data}
        layer.fused_moe = types.SimpleNamespace(
            fused_experts=types.SimpleNamespace(use_deterministic_eplb_gather=False),
            router=types.SimpleNamespace(use_deterministic_eplb_gather=False),
        )
        original = layer._expert_placement
        original_counts = layer._eplb_logic_expert_cnt.clone()
        original_inverse = layer._eplb_log2phy.clone()
        pointers = (layer.w13.data_ptr(), layer.w2.data_ptr(), layer._eplb_log2phy.data_ptr())
        placement = ExpertPlacement(2, 1, 0, (1, 0, 1, 0))
        counts, inverse = layer._routing_tensors(placement)
        layer.apply_eplb_plan(
            torch.tensor(placement.physical_to_logical, dtype=torch.int32), counts, inverse,
            {W.moe_w1: torch.zeros_like(layer.w13), W.moe_w2: torch.zeros_like(layer.w2)},
            defer_commit=True,
        )
        self.assertTrue(layer._eplb_plan_installed)
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            layer.apply_eplb_plan(
                torch.tensor(placement.physical_to_logical, dtype=torch.int32), counts, inverse, {},
            )
        layer.finalize_eplb_install(commit=False)
        self.assertEqual(layer._expert_placement, original)
        self.assertFalse(layer._eplb_plan_installed)
        self.assertFalse(layer.fused_moe.router.use_deterministic_eplb_gather)
        self.assertFalse(layer.fused_moe.fused_experts.use_deterministic_eplb_gather)
        self.assertEqual(pointers, (layer.w13.data_ptr(), layer.w2.data_ptr(), layer._eplb_log2phy.data_ptr()))
        torch.testing.assert_close(layer.w13, torch.full_like(layer.w13, 7))
        torch.testing.assert_close(layer.w2, torch.full_like(layer.w2, 8))
        torch.testing.assert_close(layer._eplb_logic_expert_cnt, original_counts)
        torch.testing.assert_close(layer._eplb_log2phy, original_inverse)

    def test_adapter_distinguishes_inherited_and_explicit_none_quant_config(self):
        source_quant = types.SimpleNamespace(get_method=lambda: "FP8")
        model_config = _runtime_model_config(2, source_quant)
        parallelism = _parallelism()
        moe_config = _moe_config()

        inherited = MoEConfigAdapter(model_config, parallelism, moe_config)
        ignored = MoEConfigAdapter(
            model_config, parallelism, moe_config, quant_config=None
        )

        self.assertIs(inherited.quant_config, source_quant)
        self.assertTrue(MoeConfigResolver.has_quantization(inherited))
        self.assertIsNone(ignored.quant_config)
        self.assertFalse(MoeConfigResolver.has_quantization(ignored))

    def test_adapter_uses_physical_expert_count_override(self):
        adapter = MoEConfigAdapter(
            _runtime_model_config(4),
            _parallelism(ep_size=2),
            _moe_config(),
            expert_num_override=6,
        )

        self.assertEqual(adapter.logical_expert_num, 4)
        self.assertEqual(adapter.expert_num, 6)

    def test_dynamic_eplb_reload_pins_cpp_worker_to_live_cuda_device(self):
        loader = NewModelLoader(
            types.SimpleNamespace(model_type="qwen_3_moe"),
            NewLoaderConfig(ep_size=2, ep_rank=1, device="cuda"),
        )
        live_experts = mock.Mock()
        live_experts.num_experts = 4
        live_experts.physical_num_experts = 6
        live_experts.w13.device = torch.device("cuda:1")
        reload_experts = mock.sentinel.reload_experts
        live_experts.clone_empty_for_placement.return_value = reload_experts

        with mock.patch.object(
            loader, "_moe_expert_for_layer", return_value=live_experts
        ), mock.patch.object(torch.cuda, "set_device") as set_device, mock.patch.object(
            loader,
            "_validate_runtime_backends",
            side_effect=RuntimeError("stop after device pin"),
        ) as validate_runtime:
            with self.assertRaisesRegex(RuntimeError, "stop after device pin"):
                loader.load_moe_layer_weights(
                    mock.sentinel.model,
                    layer_id=0,
                    physical_to_logical=[0, 1, 0, 2, 3, 3],
                    ep_rank=1,
                    ep_size=2,
                )

        set_device.assert_called_once_with(torch.device("cuda:1"))
        validate_runtime.assert_called_once_with(reload_experts, "cuda:1")

    def test_dynamic_eplb_routes_replicas_records_stats_and_installs_in_place(self):
        layer = _make_experts(
            num_experts=4,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        )
        layer._runtime_weights_dict = {
            W.moe_w1: layer.w13.data,
            W.moe_w2: layer.w2.data,
        }
        w1_ptr = layer.w13.data_ptr()
        w2_ptr = layer.w2.data_ptr()
        placement = ExpertPlacement(
            logical_expert_count=4,
            ep_size=2,
            ep_rank=0,
            physical_to_logical=(0, 1, 0, 2, 3, 3),
        )
        counts, log2phy = layer._routing_tensors(placement)
        new_w1 = torch.full_like(layer.w13, 11)
        new_w2 = torch.full_like(layer.w2, 22)
        fused_experts = types.SimpleNamespace(
            use_deterministic_eplb_gather=False
        )
        router = types.SimpleNamespace(use_deterministic_eplb_gather=False)
        layer.fused_moe = types.SimpleNamespace(
            fused_experts=fused_experts, router=router
        )

        layer.apply_eplb_plan(
            torch.tensor(placement.physical_to_logical, dtype=torch.int32),
            counts,
            log2phy,
            {W.moe_w1: new_w1, W.moe_w2: new_w2},
        )

        self.assertEqual(layer.w13.data_ptr(), w1_ptr)
        self.assertEqual(layer.w2.data_ptr(), w2_ptr)
        torch.testing.assert_close(layer.w13, new_w1)
        torch.testing.assert_close(layer.w2, new_w2)
        self.assertTrue(fused_experts.use_deterministic_eplb_gather)
        self.assertTrue(router.use_deterministic_eplb_gather)
        logical_stats = torch.zeros(4, dtype=torch.int32)
        gpu_loads = torch.zeros(2, dtype=torch.int32)
        layer.bind_eplb_stats(logical_stats, gpu_loads)
        logical_ids = torch.tensor([[0], [0], [3], [3]], dtype=torch.int32)

        physical_ids = layer._map_logical_to_physical(logical_ids)

        torch.testing.assert_close(
            physical_ids, torch.tensor([[0], [2], [4], [5]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            logical_stats, torch.tensor([2, 0, 0, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(gpu_loads, torch.tensor([2, 2], dtype=torch.int32))

    def test_dynamic_eplb_replica_selection_is_consistent_across_ep_ranks(self):
        layer = _make_experts(
            num_experts=4,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=4,
            ep_size=2,
            ep_rank=1,
            redundant_expert=2,
            enable_eplb=True,
        )
        placement = ExpertPlacement(
            logical_expert_count=4,
            ep_size=2,
            ep_rank=1,
            physical_to_logical=(0, 1, 0, 2, 3, 3),
        )
        counts, log2phy = layer._routing_tensors(placement)
        layer._expert_placement = placement
        layer._eplb_logic_expert_cnt.copy_(counts)
        layer._eplb_log2phy.copy_(log2phy)
        layer._eplb_plan_installed = True

        physical_ids = layer._map_logical_to_physical(
            torch.tensor([[0], [0], [3], [3]], dtype=torch.int32)
        )

        # Both ranks see the same logical routing decisions.  They must also
        # select the same global physical ids so exactly one owning rank
        # executes each token/expert pair before the cross-rank reduction.
        torch.testing.assert_close(
            physical_ids, torch.tensor([[0], [2], [4], [5]], dtype=torch.int32)
        )

    def test_dynamic_eplb_initial_routing_is_canonical_and_records_stats(self):
        layer = _make_experts(
            num_experts=4,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=4,
            ep_size=2,
            ep_rank=1,
            redundant_expert=2,
            enable_eplb=True,
        )
        logical_stats = torch.zeros(4, dtype=torch.int32)
        gpu_loads = torch.zeros(2, dtype=torch.int32)
        layer.bind_eplb_stats(logical_stats, gpu_loads)

        physical_ids = layer._map_logical_to_physical(
            torch.tensor([[0], [0], [3], [3]], dtype=torch.int32)
        )

        self.assertFalse(layer._eplb_plan_installed)
        torch.testing.assert_close(
            physical_ids, torch.tensor([[0], [0], [4], [4]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            logical_stats, torch.tensor([2, 0, 0, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(gpu_loads, torch.tensor([2, 2], dtype=torch.int32))

    @unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA device")
    def test_dynamic_eplb_installs_cpu_reload_into_stable_cuda_tensors(self):
        device = torch.device("cuda:0")
        layer = _make_experts(
            num_experts=4,
            top_k=1,
            hidden_size=4,
            moe_intermediate_size=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        ).to(device)
        layer._runtime_weights_dict = {
            W.moe_w1: layer.w13.data,
            W.moe_w2: layer.w2.data,
        }
        placement = ExpertPlacement(
            logical_expert_count=4,
            ep_size=2,
            ep_rank=0,
            physical_to_logical=(0, 1, 0, 2, 3, 3),
        )
        counts, log2phy = layer._routing_tensors(placement)
        w1_ptr = layer.w13.data_ptr()
        w2_ptr = layer.w2.data_ptr()

        with mock.patch.object(
            torch.cuda, "synchronize", wraps=torch.cuda.synchronize
        ) as synchronize:
            layer.apply_eplb_plan(
                torch.tensor(placement.physical_to_logical, dtype=torch.int32),
                counts,
                log2phy,
                {
                    W.moe_w1: torch.full(layer.w13.shape, 31, dtype=layer.w13.dtype),
                    W.moe_w2: torch.full(layer.w2.shape, 32, dtype=layer.w2.dtype),
                },
            )

        self.assertEqual(synchronize.call_args_list, [mock.call(device)] * 2)

        self.assertEqual(layer.w13.data_ptr(), w1_ptr)
        self.assertEqual(layer.w2.data_ptr(), w2_ptr)
        self.assertEqual(layer.w13.device, device)
        self.assertEqual(layer.w2.device, device)
        torch.testing.assert_close(layer.w13, torch.full_like(layer.w13, 31))
        torch.testing.assert_close(layer.w2, torch.full_like(layer.w2, 32))

        logical_stats = torch.zeros(4, dtype=torch.int32, device=device)
        gpu_loads = torch.zeros(2, dtype=torch.int32, device=device)
        layer.bind_eplb_stats(logical_stats, gpu_loads)
        physical_ids = layer._map_logical_to_physical(
            torch.tensor([[0], [0], [3], [3]], dtype=torch.int32, device=device)
        )
        torch.testing.assert_close(
            physical_ids.cpu(),
            torch.tensor([[0], [2], [4], [5]], dtype=torch.int32),
        )
        torch.testing.assert_close(
            logical_stats.cpu(), torch.tensor([2, 0, 0, 2], dtype=torch.int32)
        )
        torch.testing.assert_close(
            gpu_loads.cpu(), torch.tensor([2, 2], dtype=torch.int32)
        )

    def test_dynamic_eplb_rejects_inconsistent_routing_tensors(self):
        layer = _make_experts(
            num_experts=2,
            redundant_expert=2,
            enable_eplb=True,
        )
        layer._runtime_weights_dict = {
            W.moe_w1: layer.w13.data,
            W.moe_w2: layer.w2.data,
        }
        layer.w13.fill_(7)
        layer.w2.fill_(8)
        placement = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
        counts = torch.tensor([1, 3], dtype=torch.int32)
        log2phy = torch.full((2, 3), -1, dtype=torch.int32)
        old_w1 = layer.w13.clone()
        old_w2 = layer.w2.clone()
        old_placement = layer._expert_placement

        with self.assertRaisesRegex(ValueError, "do not match phy2log"):
            layer.apply_eplb_plan(
                placement,
                counts,
                log2phy,
                {W.moe_w1: layer.w13, W.moe_w2: layer.w2},
            )
        torch.testing.assert_close(layer.w13, old_w1)
        torch.testing.assert_close(layer.w2, old_w2)
        self.assertEqual(layer._expert_placement, old_placement)

    def test_newloader_eplb_controller_uses_loader_callback_contract(self):
        layer = _make_experts(
            num_experts=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        )
        model = torch.nn.Module()
        model.experts = layer
        layer._runtime_weights_dict = {
            W.moe_w1: layer.w13.data,
            W.moe_w2: layer.w2.data,
        }
        parallelism = _parallelism(ep_size=2, ep_rank=0)
        loader = types.SimpleNamespace(
            load_config=types.SimpleNamespace(parallelism_config=parallelism),
            load_moe_layer_weights=mock.Mock(
                return_value={
                    W.moe_w1: torch.ones_like(layer.w13),
                    W.moe_w2: torch.ones_like(layer.w2),
                }
            ),
        )
        model_config = types.SimpleNamespace(
            num_layers=1,
            moe_n_group=1,
            eplb_config=types.SimpleNamespace(
                balance_method="round",
                eplb_force_repack=0,
                eplb_stats_window_size=2,
            ),
        )
        controller = NewLoaderExpertBalancer(loader, model, model_config)

        plan = controller.create_balance_plan(
            torch.tensor([[9, 1, 2, 8]], dtype=torch.int32),
            torch.tensor([[10, 10]], dtype=torch.int32),
        )
        loaded = controller.load_moe_weight(plan[0], 0, 2, plan[3])

        self.assertEqual(plan[0].dtype, torch.int32)
        self.assertEqual(tuple(plan[1].shape), (4,))
        self.assertEqual(tuple(plan[2].shape), (4, 3))
        self.assertEqual(tuple(plan[3].shape), (6,))
        self.assertEqual(loaded[0], 0)
        self.assertEqual(loaded[3].numel(), 0)
        self.assertEqual(loaded[4].numel(), 0)
        loader.load_moe_layer_weights.assert_called_once_with(
            model,
            layer_id=0,
            physical_to_logical=plan[3].tolist(),
            ep_rank=0,
            ep_size=2,
        )

    def test_newloader_eplb_controller_supports_repeated_plan_install(self):
        layer = _make_experts(
            num_experts=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        )
        layer._runtime_weights_dict = {
            W.moe_w1: layer.w13.data,
            W.moe_w2: layer.w2.data,
        }
        model = torch.nn.Module()
        model.experts = layer
        parallelism = _parallelism(ep_size=2, ep_rank=0)
        loader = types.SimpleNamespace(
            load_config=types.SimpleNamespace(parallelism_config=parallelism),
            load_moe_layer_weights=mock.Mock(
                side_effect=[
                    {
                        W.moe_w1: torch.full_like(layer.w13, 11),
                        W.moe_w2: torch.full_like(layer.w2, 12),
                    },
                    {
                        W.moe_w1: torch.full_like(layer.w13, 21),
                        W.moe_w2: torch.full_like(layer.w2, 22),
                    },
                ]
            ),
        )
        model_config = types.SimpleNamespace(
            num_layers=1,
            moe_n_group=1,
            eplb_config=types.SimpleNamespace(
                balance_method="round",
                eplb_force_repack=0,
                eplb_stats_window_size=2,
            ),
        )
        controller = NewLoaderExpertBalancer(loader, model, model_config)
        w1_ptr = layer.w13.data_ptr()
        w2_ptr = layer.w2.data_ptr()

        for update, hot_expert in enumerate((0, 3), start=1):
            stats = torch.ones((1, 4), dtype=torch.int32)
            stats[0, hot_expert] = 100
            plan = controller.create_balance_plan(
                stats, torch.tensor([[100, 1]], dtype=torch.int32)
            )
            loaded = controller.load_moe_weight(plan[0], 0, 2, plan[3])
            layer.apply_eplb_plan(
                plan[3],
                plan[1],
                plan[2],
                {W.moe_w1: loaded[1], W.moe_w2: loaded[2]},
            )
            self.assertEqual(layer.w13.data_ptr(), w1_ptr)
            self.assertEqual(layer.w2.data_ptr(), w2_ptr)
            torch.testing.assert_close(
                layer.w13, torch.full_like(layer.w13, update * 10 + 1)
            )
            torch.testing.assert_close(
                layer.w2, torch.full_like(layer.w2, update * 10 + 2)
            )
            self.assertEqual(
                controller.phy2log[0],
                list(layer._expert_placement.physical_to_logical),
            )

        self.assertEqual(loader.load_moe_layer_weights.call_count, 2)

    def test_newloader_eplb_controller_rolls_back_failed_load_state(self):
        layer = _make_experts(
            num_experts=4,
            ep_size=2,
            ep_rank=0,
            redundant_expert=2,
            enable_eplb=True,
        )
        model = torch.nn.Module()
        model.experts = layer
        parallelism = _parallelism(ep_size=2, ep_rank=0)
        loader = types.SimpleNamespace(
            load_config=types.SimpleNamespace(parallelism_config=parallelism),
            load_moe_layer_weights=mock.Mock(
                side_effect=RuntimeError("injected checkpoint read failure")
            ),
        )
        model_config = types.SimpleNamespace(
            num_layers=1,
            moe_n_group=1,
            eplb_config=types.SimpleNamespace(
                balance_method="round",
                eplb_force_repack=0,
                eplb_stats_window_size=2,
            ),
        )
        controller = NewLoaderExpertBalancer(loader, model, model_config)
        installed = list(layer._expert_placement.physical_to_logical)
        planned = torch.tensor([0, 1, 2, 3, 3, 2], dtype=torch.int32)

        with self.assertRaisesRegex(RuntimeError, "injected checkpoint read failure"):
            controller.load_moe_weight(
                torch.tensor([0], dtype=torch.int32), 0, 2, planned
            )
        self.assertEqual(controller.phy2log[0], planned.tolist())

        controller.rollback_balance_plan(0)

        self.assertEqual(controller.phy2log[0], installed)
        self.assertEqual(list(layer._expert_placement.physical_to_logical), installed)


class Qwen3MoeModelTest(unittest.TestCase):
    def _config(self, tie_word_embeddings=True):
        config = ModelConfig()
        config.model_type = "qwen_3_moe"
        config.num_layers = 1
        config.vocab_size = 8
        config.hidden_size = 4
        config.inter_size = 4
        config.expert_num = 2
        config.moe_inter_size = 4
        config.moe_k = 1
        config.moe_topk_group = 1
        config.attn_config.head_num = 2
        config.attn_config.kv_head_num = 1
        config.attn_config.size_per_head = 2
        config.layernorm_eps = 1e-6
        config.enable_fp32_lm_head = False
        config.tie_word_embeddings = tie_word_embeddings
        config.data_type = "fp32"
        config.quant_config = None
        config.activation_type = "SiGLU"
        return config

    def _load_config(self, tp_size=1, tp_rank=0, ep_size=1, ep_rank=0):
        parallelism = _parallelism(tp_size, tp_rank, ep_size, ep_rank)
        return NewLoaderConfig(
            tp_size=tp_size,
            tp_rank=tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            compute_dtype=torch.float32,
            device="cpu",
            quant_config=QuantizationConfig("none"),
            parallelism_config=parallelism,
            moe_config=_moe_config(),
        )

    def _weights(self):
        weights = {
            "model.embed_tokens.weight": torch.arange(32, dtype=torch.float32).reshape(
                8, 4
            ),
            "model.layers.0.input_layernorm.weight": torch.ones(4),
            "model.layers.0.self_attn.q_proj.weight": torch.ones(4, 4),
            "model.layers.0.self_attn.k_proj.weight": torch.ones(2, 4),
            "model.layers.0.self_attn.v_proj.weight": torch.ones(2, 4),
            "model.layers.0.self_attn.q_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.k_norm.weight": torch.ones(2),
            "model.layers.0.self_attn.o_proj.weight": torch.eye(4),
            "model.layers.0.post_attention_layernorm.weight": torch.ones(4),
            "model.layers.0.mlp.gate.weight": torch.ones(2, 4),
            "model.norm.weight": torch.ones(4),
        }
        for name, tensor in _expert_weights(2, 4, 4).items():
            weights[f"model.layers.0.mlp.experts.{name}"] = tensor
        return weights

    def test_real_model_tree_loads_and_ties_lm_head(self):
        model = Qwen3MoeForCausalLM(self._config(), self._load_config())
        model.load_weights(self._weights())
        NewModelLoader._validate_loaded_weights(model)

        torch.testing.assert_close(model.lm_head.weight, model.embed_tokens.weight)
        self.assertEqual(model.layers[0].mlp.experts._loaded_count, 6)
        self.assertEqual(
            set(model.runtime_weight_view()),
            {"embedding", "final_layernorm.gamma", "lm_head"},
        )

    def test_hf_dict_is_rejected_before_router_construction(self):
        config = {
            "model_type": "qwen_3_moe",
            "num_hidden_layers": 1,
            "vocab_size": 8,
            "hidden_size": 4,
            "intermediate_size": 0,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 2,
            "rms_norm_eps": 1e-6,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "moe_intermediate_size": 4,
            "norm_topk_prob": True,
            "tie_word_embeddings": True,
        }

        with self.assertRaisesRegex(TypeError, "requires a typed ModelConfig"):
            Qwen3MoeForCausalLM(config, self._load_config())

    def test_real_select_topk_receives_runtime_model_config(self):
        config = self._config()

        model = Qwen3MoeForCausalLM(config, self._load_config())

        self.assertIs(model.layers[0].mlp.select_topk.config, config)

    def test_router_config_rejects_invalid_normalization_flag(self):
        config = self._config()
        with self.assertRaises(TypeError):
            config.has_moe_norm = "true"

    def test_zero_dense_intermediate_size_uses_moe_intermediate_size(self):
        config = self._config()
        config.inter_size = 0

        model = Qwen3MoeForCausalLM(config, self._load_config())

        self.assertEqual(model.layers[0].mlp.experts.moe_inter, 4)

    def test_eplb_attribute_enables_dynamic_newloader_moe(self):
        config = self._config()
        config.eplb_config.eplb_mode = EplbMode.EPLB

        model = Qwen3MoeForCausalLM(config, self._load_config())

        self.assertTrue(model.layers[0].mlp.experts._dynamic_eplb_enabled)

    def test_invalid_adapter_cuda_graph_flag_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "enable_cuda_graph must be a bool"):
            MoEConfigAdapter(
                self._config(),
                _parallelism(),
                _moe_config(),
                enable_cuda_graph=1,
            )

    def test_w4a8_source_config_rejects_invalid_dimensions(self):
        from rtp_llm.config.quant_config import (
            CompressedW4A8Int4PerChannelQuantConfig,
            W4a8Int4PerChannelQuantConfig,
        )

        for config_cls in (
            W4a8Int4PerChannelQuantConfig,
            CompressedW4A8Int4PerChannelQuantConfig,
        ):
            for kwargs in (
                {"bits": 8, "group_size": 128},
                {"bits": 4, "group_size": 0},
                {"bits": True, "group_size": 128},
            ):
                with self.subTest(config=config_cls.__name__, kwargs=kwargs):
                    with self.assertRaises((TypeError, ValueError)):
                        config_cls(is_quanted=True, **kwargs)

    def test_source_quant_configs_preserve_normalized_ignore_patterns(self):
        from rtp_llm.config.quant_config import (
            CompressedW4A8Int4PerChannelQuantConfig,
            ModelOptFp4Config,
            MXFp4QuarkQuantConfig,
        )

        ignored = [" model.layers.0.mlp ", "model.layers.0.mlp"]
        for config in (
            MXFp4QuarkQuantConfig(
                bits=4,
                group_size=0,
                is_quanted=True,
                ignored_layers=ignored,
            ),
            ModelOptFp4Config(
                bits=4,
                group_size=16,
                is_quanted=True,
                ignored_layers=ignored,
            ),
        ):
            self.assertEqual(config.ignored_layers, ["model.layers.0.mlp"])

        compressed = CompressedW4A8Int4PerChannelQuantConfig(
            bits=4,
            group_size=32,
            is_quanted=True,
            ignore_patterns=ignored,
        )
        self.assertIs(compressed.ignore_patterns, compressed.ignored_layers)
        self.assertEqual(compressed.ignore_patterns, ["model.layers.0.mlp"])

    def test_modelopt_checkpoint_preserves_ignore_and_exclude_patterns(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "config.json"), "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "modelopt",
                            "ignore": [" model.layers.0.mlp "],
                            "exclude": ["lm_head"],
                            "config_groups": {
                                "group_0": {
                                    "weights": {
                                        "type": "float",
                                        "num_bits": 4,
                                        "group_size": 16,
                                    },
                                    "input_activations": {"num_bits": 4},
                                }
                            },
                        }
                    },
                    output,
                )

            parsed = SourceQuantizationConfig.load_from_ckpt(model_path)

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.ignored_layers, ["model.layers.0.mlp"])
        self.assertEqual(parsed.exclude_modules, {"lm_head"})

    def test_invalid_boolean_config_is_rejected(self):
        config = self._config()
        config.tie_word_embeddings = "true"
        with self.assertRaisesRegex(TypeError, "tie_word_embeddings"):
            Qwen3MoeForCausalLM(config, self._load_config())

        config = self._config()
        load_config = self._load_config()
        load_config.moe_config.fake_balance_expert = 1
        with self.assertRaisesRegex(TypeError, "fake_balance_expert"):
            Qwen3MoeForCausalLM(config, load_config)

    def test_untied_missing_lm_head_fails_integrity(self):
        model = Qwen3MoeForCausalLM(
            self._config(tie_word_embeddings=False), self._load_config()
        )
        model.load_weights(self._weights())
        with self.assertRaisesRegex(RuntimeError, "ParallelLMHead.*weight"):
            NewModelLoader._validate_loaded_weights(model)

    def test_tp_owners_use_reviewed_topology(self):
        model = Qwen3MoeForCausalLM(
            self._config(), self._load_config(tp_size=2, tp_rank=1)
        )
        self.assertEqual(
            (model.embed_tokens.tp_size, model.embed_tokens.tp_rank), (2, 1)
        )
        self.assertEqual(
            (
                model.layers[0].self_attn.qkv_proj.tp_size,
                model.layers[0].self_attn.qkv_proj.tp_rank,
            ),
            (2, 1),
        )
        experts = model.layers[0].mlp.experts
        self.assertEqual(
            (experts.moe_expert_tp_size, experts.moe_expert_tp_rank), (2, 1)
        )
        self.assertEqual((model.lm_head.tp_size, model.lm_head.tp_rank), (2, 1))

    def test_router_gate_stays_unquantized_for_fp8_model(self):
        source_quant = types.SimpleNamespace(
            get_runtime_method_key=lambda: "FP8_PER_BLOCK",
            get_method=lambda: "FP8_PER_BLOCK",
            weight_block_size=[128, 128],
            modules_to_not_convert=[],
        )
        config = self._config()
        config.quant_config = source_quant
        load_config = self._load_config()
        load_config = types.SimpleNamespace(
            **{
                **load_config.__dict__,
                "quant_config": QuantizationConfig(
                    "FP8_PER_BLOCK",
                    source_config=source_quant,
                ),
            }
        )

        model = Qwen3MoeForCausalLM(config, load_config)

        self.assertEqual(
            type(model.layers[0].mlp.gate.quant_method).__name__,
            "UnquantizedLinearMethod",
        )
        self.assertEqual(model.layers[0].mlp.experts._quant_family, "fp8_per_block")

    def test_checkpoint_modules_to_not_convert_reaches_runtime_dispatch(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "config.json"), "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "fp8",
                            "weight_block_size": [128, 128],
                            "modules_to_not_convert": [
                                "model.layers.0.mlp.gate",
                                "lm_head",
                            ],
                        }
                    },
                    output,
                )

            source_quant = SourceQuantizationConfig.load_from_ckpt(model_path)

        self.assertIsNotNone(source_quant)
        runtime_quant = QuantizationConfig(
            source_quant.get_runtime_method_key(), source_config=source_quant
        )
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.mlp.gate"))
        self.assertTrue(runtime_quant.is_layer_ignored("lm_head"))
        self.assertFalse(
            runtime_quant.is_layer_ignored("layers.0.mlp.experts.gate_proj")
        )

    def test_compressed_w4a8_json_preserves_all_ignore_aliases(self):
        with tempfile.TemporaryDirectory() as model_path:
            with open(os.path.join(model_path, "config.json"), "w") as output:
                json.dump(
                    {
                        "quantization_config": {
                            "quant_method": "compressed-tensors",
                            "ignore": ["model.layers.{i}.self_attn"],
                            "modules_to_not_convert": ["model.layers.{i}.mlp.gate"],
                            "config_groups": {
                                "group_0": {
                                    "weights": {
                                        "type": "int",
                                        "num_bits": 4,
                                        "strategy": "group",
                                        "group_size": 32,
                                    },
                                    "input_activations": {
                                        "type": "float",
                                        "num_bits": 8,
                                        "dynamic": True,
                                    },
                                }
                            },
                        }
                    },
                    output,
                )

            source_quant = SourceQuantizationConfig.load_from_ckpt(model_path)

        self.assertIsNotNone(source_quant)
        runtime_quant = QuantizationConfig(
            source_quant.get_runtime_method_key(), source_config=source_quant
        )
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.self_attn.q_proj"))
        self.assertTrue(runtime_quant.is_layer_ignored("layers.0.mlp.gate"))
        self.assertFalse(
            runtime_quant.is_layer_ignored("layers.0.mlp.experts.gate_proj")
        )


if __name__ == "__main__":
    unittest.main()
