import random
import unittest
from typing import Tuple

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.kernels.cuda.fp8_kernel.fp8_kernel import (
    per_block_cast_to_fp8,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    ExpertForwardPayload,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_fused_moe import (
    CutlassFusedMoeFp8PerBlockExecutor,
)
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.utils.model_weight import W


def _ref_fp8_block_moe(
    hidden_states_bf16: torch.Tensor,
    w1_fp8: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_fp8: torch.Tensor,
    w2_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    block_size: int = 128,
) -> torch.Tensor:
    """Reference SiLU-and-mul MoE computed in BF16 from dequantized weights."""
    num_tokens, hidden_size = hidden_states_bf16.shape
    num_experts = w1_fp8.shape[0]
    inter_size_2 = w1_fp8.shape[1]
    inter_size = inter_size_2 // 2
    top_k = topk_ids.shape[1]

    def dequant(weight_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        # weight_fp8: (E, N, K), scale: (E, N//block, K//block)
        e, n, k = weight_fp8.shape
        w_fp32 = weight_fp8.to(torch.float32)
        scale_expand = (
            scale.repeat_interleave(block_size, dim=1)
            .repeat_interleave(block_size, dim=2)[:, :n, :k]
        )
        return (w_fp32 * scale_expand).to(torch.bfloat16)

    w1_bf16 = dequant(w1_fp8, w1_scale)
    w2_bf16 = dequant(w2_fp8, w2_scale)

    output = torch.zeros_like(hidden_states_bf16)
    for token_idx in range(num_tokens):
        x = hidden_states_bf16[token_idx]
        token_out = torch.zeros(hidden_size, dtype=torch.float32, device=x.device)
        for k in range(top_k):
            expert_id = int(topk_ids[token_idx, k].item())
            weight = float(topk_weights[token_idx, k].item())
            if expert_id < 0 or expert_id >= num_experts:
                continue
            gate_up = x.to(torch.float32) @ w1_bf16[expert_id].t().to(torch.float32)
            gate = gate_up[:inter_size]
            up = gate_up[inter_size:]
            # cutlass swiglu: silu(first_half) * second_half
            act = (gate * torch.sigmoid(gate)) * up
            down = act @ w2_bf16[expert_id].t().to(torch.float32)
            token_out += weight * down
        output[token_idx] = token_out.to(torch.bfloat16)
    return output


def _maybe_skip_unsupported() -> Tuple[bool, str]:
    if not torch.cuda.is_available():
        return True, "CUDA not available"
    sm_major = get_sm()[0]
    if sm_major != 9:
        return True, f"cutlass_fused_moe FP8 block scale only supported on SM90 (got SM{sm_major}.x)"
    try:
        import flashinfer.fused_moe  # noqa: F401
    except ImportError as e:
        return True, f"flashinfer.fused_moe unavailable: {e}"
    return False, ""


class CutlassFusedMoeFp8PerBlockExecutorTest(unittest.TestCase):
    NUM_EXPERTS = 16
    HIDDEN_SIZE = 256
    MOE_INTERMEDIATE_SIZE = 256
    NUM_TOKENS = 32
    TOP_K = 4

    def setUp(self):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        random.seed(0)

    def _build_config(self) -> MoEConfigAdapter:
        model_config = ModelConfig()
        model_config.attn_config.head_num = 2
        model_config.attn_config.size_per_head = 128
        model_config.num_layers = 2
        model_config.max_seq_len = 2048
        model_config.vocab_size = 1000
        model_config.expert_num = self.NUM_EXPERTS
        model_config.hidden_size = self.HIDDEN_SIZE
        model_config.moe_inter_size = self.MOE_INTERMEDIATE_SIZE
        model_config.moe_k = self.TOP_K

        parallelism_config = ParallelismConfig()
        parallelism_config.world_size = 1
        parallelism_config.dp_size = 1
        parallelism_config.tp_size = 1
        parallelism_config.ep_size = 1
        parallelism_config.dp_rank = 0
        parallelism_config.tp_rank = 0
        parallelism_config.ep_rank = 0
        parallelism_config.world_rank = 0
        parallelism_config.local_world_size = 1

        moe_config = MoeConfig()
        moe_config.moe_strategy = "fp8_per_block_no_dp_cutlass_fused"
        moe_config.use_all_gather = True
        return MoEConfigAdapter(
            model_config=model_config,
            parallelism_config=parallelism_config,
            moe_config=moe_config,
        )

    def _make_weights(self):
        E = self.NUM_EXPERTS
        K = self.HIDDEN_SIZE
        N2 = self.MOE_INTERMEDIATE_SIZE * 2  # gate + up

        w1_bf16 = (
            torch.randn((E, N2, K), device="cuda", dtype=torch.float32) * 0.05
        ).to(torch.bfloat16)
        w2_bf16 = (
            torch.randn((E, K, N2 // 2), device="cuda", dtype=torch.float32) * 0.05
        ).to(torch.bfloat16)

        w1_fp8 = torch.zeros_like(w1_bf16, dtype=torch.float8_e4m3fn)
        w1_scale = torch.zeros(
            (E, N2 // 128, K // 128), device="cuda", dtype=torch.float32
        )
        w2_fp8 = torch.zeros_like(w2_bf16, dtype=torch.float8_e4m3fn)
        w2_scale = torch.zeros(
            (E, K // 128, (N2 // 2) // 128), device="cuda", dtype=torch.float32
        )
        for i in range(E):
            w1_fp8[i], w1_scale[i] = per_block_cast_to_fp8(
                w1_bf16[i], use_ue8m0=False
            )
            w2_fp8[i], w2_scale[i] = per_block_cast_to_fp8(
                w2_bf16[i], use_ue8m0=False
            )
        return w1_fp8, w1_scale, w2_fp8, w2_scale

    def test_executor_matches_reference(self):
        skip, reason = _maybe_skip_unsupported()
        if skip:
            self.skipTest(reason)

        # Validate shape constraints we rely on (multiples of 128)
        assert self.HIDDEN_SIZE % 128 == 0
        assert self.MOE_INTERMEDIATE_SIZE % 128 == 0

        config = self._build_config()
        w1_fp8, w1_scale, w2_fp8, w2_scale = self._make_weights()

        hidden_states = (
            torch.randn(
                (self.NUM_TOKENS, self.HIDDEN_SIZE),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.5
        ).to(torch.bfloat16)

        # Random topk routing.
        logits = torch.randn(
            (self.NUM_TOKENS, self.NUM_EXPERTS),
            device="cuda",
            dtype=torch.float32,
        )
        topk_weights, topk_ids = torch.topk(
            torch.softmax(logits, dim=-1), self.TOP_K, dim=-1
        )
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(torch.float32)

        weights = {
            W.moe_w1: w1_fp8,
            W.moe_w2: w2_fp8,
            W.moe_s1: w1_scale,
            W.moe_s2: w2_scale,
        }

        executor = CutlassFusedMoeFp8PerBlockExecutor(
            config,
            FusedMoEQuantConfig(
                quant_dtype=torch.float8_e4m3fn,
                block_shape=[128, 128],
            ),
            weights,
        )
        payload = ExpertForwardPayload(
            expert_x=hidden_states,
            expert_x_origin_dtype=torch.bfloat16,
            expert_x_scale=None,
            expert_topk_ids=topk_ids,
            expert_topk_weights=topk_weights,
        )

        combine = executor.execute(payload, "SiGLU", None, None, False, None)
        actual = combine.fused_expert_output

        ref = _ref_fp8_block_moe(
            hidden_states,
            w1_fp8,
            w1_scale,
            w2_fp8,
            w2_scale,
            topk_ids,
            topk_weights,
        )

        self.assertEqual(actual.shape, hidden_states.shape)
        self.assertEqual(actual.dtype, torch.bfloat16)
        # FP8-block-quant + per-token quant inside the kernel introduces noise;
        # require a coarse-but-meaningful match.
        diff = (actual.float() - ref.float()).abs()
        max_abs = diff.max().item()
        rel = (diff / (ref.float().abs() + 1e-3)).mean().item()
        self.assertLess(
            max_abs,
            5e-1,
            f"max abs diff too large: max={max_abs}, rel_mean={rel}",
        )


if __name__ == "__main__":
    unittest.main()
