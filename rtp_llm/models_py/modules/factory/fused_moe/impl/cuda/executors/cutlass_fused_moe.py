"""CUDA executors that wrap flashinfer.fused_moe.cutlass_fused_moe.

Restricted to:
  * SM90 (Hopper)
  * FP8 PerBlock weights with deepseek-style 128x128 block scaling

Three variants are provided, distinguished by the router that feeds them:
  * CutlassFusedMoeFp8PerBlockExecutor — pure-TP / single-card (PureTpRouterFp8PerBlockBf16Passthrough)
  * CutlassFusedMoeFp8PerBlockEpLowLatencyExecutor — DeepEP low-latency masked dispatch
  * CutlassFusedMoeFp8PerBlockEpNormalExecutor — DeepEP normal token-level dispatch

The kernel internally re-quantizes the BF16 input to FP8 (1x128 tile) and
re-uses the existing per-block FP8 weights / scales loaded by RTP-LLM, so the
EP variants dequantize the FP8 dispatch output back to BF16 before the call.
"""

from typing import Any, Dict, Optional

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.fused_moe import (
    CombineForwardPayload,
    ExpertForwardPayload,
    FusedMoeExpertExecutor,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.type import ExecutorType
from rtp_llm.models_py.utils.arch import get_sm
from rtp_llm.models_py.utils.memory import dispose_tensor
from rtp_llm.utils.model_weight import W


_FP8_BLOCK_K = 128


def _check_cutlass_fp8_per_block_conditions(
    checker: Any, config: MoEConfigAdapter
) -> None:
    from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
        MoeConfigResolver,
    )

    resolver = MoeConfigResolver()
    quant_method = resolver.get_quant_method(config)
    checker.check(quant_method == "FP8_PER_BLOCK")
    checker.check(get_sm()[0] == 9)
    try:
        import flashinfer.fused_moe  # noqa: F401
    except ImportError:
        checker.check(False)
        return
    checker.check(True)


def _dequant_fp8_per_block_to_bf16(
    x_fp8: torch.Tensor, x_scale: torch.Tensor
) -> torch.Tensor:
    """Dequantize per-token-group FP8 (block_k=128) activations to BF16.

    Args:
        x_fp8: FP8 tensor with shape [..., K] where K % 128 == 0.
        x_scale: float32 scale tensor with shape [..., K // 128].

    Returns:
        BF16 tensor with the same shape as x_fp8.
    """
    last_k = x_fp8.shape[-1]
    assert last_k % _FP8_BLOCK_K == 0, f"K={last_k} must be a multiple of {_FP8_BLOCK_K}"
    scale_expanded = x_scale.repeat_interleave(_FP8_BLOCK_K, dim=-1)[..., :last_k]
    return (x_fp8.to(torch.float32) * scale_expanded.to(torch.float32)).to(
        torch.bfloat16
    )


class _CutlassFp8PerBlockBase(FusedMoeExpertExecutor):
    """Shared weight setup for cutlass_fused_moe FP8 PerBlock executors."""

    @classmethod
    def executor_type(cls) -> ExecutorType:
        return ExecutorType.CUTLASS_FP8

    @property
    def topk_ids_dtype(self) -> torch.dtype:
        return torch.int32

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        _check_cutlass_fp8_per_block_conditions(checker, config)

    def __init__(
        self,
        config: MoEConfigAdapter,
        quant_config: FusedMoEQuantConfig,
        weights: Dict[str, torch.Tensor],
    ):
        super().__init__(config, quant_config, weights)

        self.num_experts = config.expert_num
        self.tp_size = config.tp_size
        self.tp_rank = config.tp_rank
        self.ep_size = config.ep_size
        self.ep_rank = config.ep_rank
        self.top_k = config.moe_k
        self.expert_num_per_rank = self.num_experts // max(self.ep_size, 1)

        self.w13_weight = weights[W.moe_w1]
        self.w2_weight = weights[W.moe_w2]
        self.w13_weight_scale_inv = weights[W.moe_s1].contiguous()
        self.w2_weight_scale_inv = weights[W.moe_s2].contiguous()

        assert (
            self.w13_weight.dtype == torch.float8_e4m3fn
        ), f"w13_weight dtype must be float8_e4m3fn, got {self.w13_weight.dtype}"
        assert (
            self.w2_weight.dtype == torch.float8_e4m3fn
        ), f"w2_weight dtype must be float8_e4m3fn, got {self.w2_weight.dtype}"
        assert (
            self.w13_weight_scale_inv.dtype == torch.float32
        ), f"w13_weight_scale_inv dtype must be float32, got {self.w13_weight_scale_inv.dtype}"
        assert (
            self.w2_weight_scale_inv.dtype == torch.float32
        ), f"w2_weight_scale_inv dtype must be float32, got {self.w2_weight_scale_inv.dtype}"

    def _call_cutlass_fused_moe(
        self,
        hidden_states_bf16: torch.Tensor,
        topk_ids_i32: torch.Tensor,
        topk_weights_f32: torch.Tensor,
        output_dtype: torch.dtype,
        ep_size: int,
        ep_rank: int,
    ) -> torch.Tensor:
        from flashinfer.fused_moe import cutlass_fused_moe

        quant_scales = [self.w13_weight_scale_inv, self.w2_weight_scale_inv]
        result = cutlass_fused_moe(
            input=hidden_states_bf16,
            token_selected_experts=topk_ids_i32,
            token_final_scales=topk_weights_f32,
            fc1_expert_weights=self.w13_weight,
            fc2_expert_weights=self.w2_weight,
            output_dtype=output_dtype,
            quant_scales=quant_scales,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            use_deepseek_fp8_block_scale=True,
        )
        if isinstance(result, (list, tuple)):
            return result[0]
        return result


class CutlassFusedMoeFp8PerBlockExecutor(_CutlassFp8PerBlockBase):
    """Wraps flashinfer.fused_moe.cutlass_fused_moe for FP8 per-block MoE.

    Pairs with PureTpRouterFp8PerBlockBf16Passthrough: receives BF16 input
    unmodified and routes the global topk decisions directly into the kernel.
    """

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert payload.expert_x is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        hidden_states = payload.expert_x
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        assert hidden_states.dtype in (
            torch.bfloat16,
            torch.float16,
            torch.float32,
        ), (
            "cutlass_fused_moe (FP8 block scale) expects unquantized hidden states; "
            f"got {hidden_states.dtype}"
        )
        assert payload.expert_x_scale is None
        assert activation == "SiGLU"

        output_dtype = (
            payload.expert_x_origin_dtype
            if payload.expert_x_origin_dtype is not None
            else hidden_states.dtype
        )

        topk_ids_i32 = (
            topk_ids
            if topk_ids.dtype == torch.int32
            else topk_ids.to(torch.int32)
        )
        topk_weights_f32 = (
            topk_weights
            if topk_weights.dtype == torch.float32
            else topk_weights.to(torch.float32)
        )

        output = self._call_cutlass_fused_moe(
            hidden_states_bf16=hidden_states,
            topk_ids_i32=topk_ids_i32,
            topk_weights_f32=topk_weights_f32,
            output_dtype=output_dtype,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
        )
        return CombineForwardPayload(fused_expert_output=output)


class CutlassFusedMoeFp8PerBlockEpLowLatencyExecutor(_CutlassFp8PerBlockBase):
    """Cutlass FP8 PerBlock executor for DeepEP low-latency masked dispatch.

    DeepEP low-latency emits a 3D masked tensor [E_local, M, K] (FP8) plus a
    per-expert valid-token count. flashinfer's cutlass_fused_moe wants a flat
    2D [num_tokens, K] BF16 input with explicit per-row expert IDs. We adapt:
      1. Dequantize FP8 -> BF16 block-wise.
      2. Flatten to [E_local*M, K].
      3. Synthesize per-row expert IDs (row e*M+m -> expert e).
      4. Mask padded rows (m >= expert_num_tokens[e]) by setting weight=0.
      5. Reshape kernel output back to [E_local, M, K] for the masked combine.
    """

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert activation == "SiGLU"
        assert payload.expert_x is not None
        assert payload.expert_x_scale is not None
        assert payload.expert_tokens_meta is not None
        assert payload.expert_tokens_meta.expert_num_tokens is not None

        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale
        expert_num_tokens = payload.expert_tokens_meta.expert_num_tokens

        assert expert_x.dim() == 3, (
            f"DeepEP low-latency payload expects [E_local, M, K]; got {tuple(expert_x.shape)}"
        )
        e_local, m_max, k = expert_x.shape
        assert e_local == self.expert_num_per_rank, (
            f"E_local={e_local} != expert_num_per_rank={self.expert_num_per_rank}"
        )
        assert expert_x_scale.shape[:2] == (e_local, m_max)
        assert expert_x_scale.shape[-1] == k // _FP8_BLOCK_K

        device = expert_x.device
        output_dtype = (
            payload.expert_x_origin_dtype
            if payload.expert_x_origin_dtype is not None
            else torch.bfloat16
        )

        bf16_3d = _dequant_fp8_per_block_to_bf16(expert_x, expert_x_scale)
        dispose_tensor(expert_x)
        dispose_tensor(expert_x_scale)

        flat_input = bf16_3d.reshape(e_local * m_max, k)

        expert_ids = (
            torch.arange(e_local, device=device, dtype=torch.int32)
            .view(e_local, 1)
            .expand(e_local, m_max)
            .reshape(-1, 1)
            .contiguous()
        )

        token_index = torch.arange(m_max, device=device).unsqueeze(0)
        valid_mask = token_index < expert_num_tokens.to(torch.int64).unsqueeze(1)
        weights = valid_mask.to(torch.float32).reshape(-1, 1).contiguous()

        # ep_size=1: fc1/fc2 weights already contain only the local experts and
        # the synthesized expert IDs are local-relative.
        flat_output = self._call_cutlass_fused_moe(
            hidden_states_bf16=flat_input,
            topk_ids_i32=expert_ids,
            topk_weights_f32=weights,
            output_dtype=output_dtype,
            ep_size=1,
            ep_rank=0,
        )

        return CombineForwardPayload(
            fused_expert_output=flat_output.view(e_local, m_max, k)
        )


class CutlassFusedMoeFp8PerBlockEpNormalExecutor(_CutlassFp8PerBlockBase):
    """Cutlass FP8 PerBlock executor for DeepEP normal token-level dispatch.

    DeepepNormalRouterFp8PerBlock emits a flat [total_tokens, K] FP8 tensor
    plus per-block scales, with topk IDs already in local [0, E_local) space
    (the base router only shifts to global when use_fp8 is False). We
    dequantize to BF16 and call cutlass_fused_moe with ep_size=1 since the
    weights and IDs are both local.
    """

    def execute(
        self,
        payload: ExpertForwardPayload,
        activation: str,
        expert_map: Optional[torch.Tensor],
        a2_scale: Optional[torch.Tensor],
        apply_router_weight_on_input: bool,
        extra_expert_args: Optional[dict[str, Any]],
    ) -> CombineForwardPayload:
        assert activation == "SiGLU"
        assert payload.expert_x is not None
        assert payload.expert_x_scale is not None
        assert payload.expert_topk_ids is not None
        assert payload.expert_topk_weights is not None

        expert_x = payload.expert_x
        expert_x_scale = payload.expert_x_scale
        topk_ids = payload.expert_topk_ids
        topk_weights = payload.expert_topk_weights

        assert expert_x.dim() == 2, (
            f"DeepEP normal payload expects [total_tokens, K]; got {tuple(expert_x.shape)}"
        )
        total_tokens, k = expert_x.shape
        assert expert_x_scale.shape[0] == total_tokens
        assert expert_x_scale.shape[-1] == k // _FP8_BLOCK_K

        output_dtype = (
            payload.expert_x_origin_dtype
            if payload.expert_x_origin_dtype is not None
            else torch.bfloat16
        )

        bf16_input = _dequant_fp8_per_block_to_bf16(expert_x, expert_x_scale)
        dispose_tensor(expert_x)
        dispose_tensor(expert_x_scale)

        topk_ids_i32 = (
            topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        )
        topk_weights_f32 = (
            topk_weights
            if topk_weights.dtype == torch.float32
            else topk_weights.to(torch.float32)
        )

        # DeepEP marks unrouted slots with -1; clamp to a valid local id and
        # zero out the corresponding weight so they contribute nothing.
        invalid_mask = topk_ids_i32 < 0
        if invalid_mask.any():
            topk_ids_i32 = torch.where(
                invalid_mask, torch.zeros_like(topk_ids_i32), topk_ids_i32
            )
            topk_weights_f32 = torch.where(
                invalid_mask, torch.zeros_like(topk_weights_f32), topk_weights_f32
            )

        output = self._call_cutlass_fused_moe(
            hidden_states_bf16=bf16_input,
            topk_ids_i32=topk_ids_i32,
            topk_weights_f32=topk_weights_f32,
            output_dtype=output_dtype,
            ep_size=1,
            ep_rank=0,
        )
        return CombineForwardPayload(fused_expert_output=output)
