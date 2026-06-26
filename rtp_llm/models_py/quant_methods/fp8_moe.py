"""FP8 MoE 量化方法（vLLM 风格,从 BaseMoEExperts 内置 fp8 逻辑迁出）。

覆盖全部 fp8 子族(已量化:per_tensor/per_channel/per_block;在线 BF16->FP8:三者的
*_online)。逻辑与 `BaseMoEExperts` 内置 fp8 路径**逐字一致**(只把 ``self.`` 换成
``layer.``),按 ``layer._quant_family`` 分支:
  - ``create_weights``  ← `_init_buffers` 的 fp8/online 分支
  - ``dispatch_scale``  ← `_dispatch_scale` + `_copy_per_channel_scale` + `_copy_block_scale`
  - ``process_weights_after_loading`` ← `_fuse_fp8_*` / `_online_quantize_*`
  - ``add_weight_tensors`` ← `_build_weights_dict` 的 fp8 scale 注入
前向仍由 `BaseMoEExperts.forward` 经 fused_moe 完成。

验证状态:per_tensor 已端到端验证;per_channel/per_block/online 为忠实搬移,各需对应
ckpt 验证。回退某子族:从下面 `@register_moe_quant_method` 移除对应 key → 走内置。
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from rtp_llm.models_py.quant_methods.base import (
    FusedMoEMethodBase,
    register_moe_quant_method,
)
from rtp_llm.utils.model_weight import W

# 与 BaseMoEExperts 一致的 fp8 常量（也可经 layer 取，这里就近定义保持自洽）。
_FP8_E4M3_MAX: float = 448.0
_FP8_MIN_SCALE: float = 1.0 / (448.0 * 512.0)


@register_moe_quant_method(
    # per_tensor 子族
    "fp8",
    "FP8_PER_TENSOR_COMPRESSED",
    "FP8_DYNAMIC_PER_TENSOR",
    # per_block 子族
    "FP8_PER_BLOCK",
    "fp8_block",
    # per_channel 子族
    "FP8_PER_CHANNEL_COMPRESSED",
    "fp8_per_channel",
    "FP8_PER_CHANNEL_QUARK",
    # 在线 BF16->FP8 子族
    "fp8_online",
    "fp8_block_online",
    "fp8_per_channel_online",
)
class Fp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: Any = None):
        self.quant_config = quant_config

    # ------------------------------------------------------------------ #
    #  create_weights ← _init_buffers 的 fp8/online 分支
    # ------------------------------------------------------------------ #
    def create_weights(
        self,
        layer,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: torch.dtype,
        **kwargs,
    ):
        E, M_tp, H = num_experts, intermediate_size, hidden_size
        qf = layer._quant_family
        dt = torch.float8_e4m3fn
        BS = layer._FP8_BLOCK_SIZE

        if qf == "fp8_per_tensor":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer("_gate_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_up_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("_down_scales", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
            layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))

        elif qf == "fp8_per_channel":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            layer.register_buffer(
                "_gate_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_up_ch_scales", torch.zeros(E, M_tp, dtype=torch.float32)
            )
            layer.register_buffer(
                "_down_ch_scales", torch.zeros(E, H, dtype=torch.float32)
            )
            layer.register_buffer(
                "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
            )
            layer.register_buffer("w2_scale", torch.zeros(E, H, dtype=torch.float32))

        elif qf == "fp8_per_block":
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=dt), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=dt), requires_grad=False
            )
            nb = (M_tp + BS - 1) // BS
            layer._n_scale_blocks_per_proj = nb
            layer.register_buffer(
                "w13_scale",
                torch.zeros(E, 2 * nb, (H + BS - 1) // BS, dtype=torch.float32),
            )
            layer.register_buffer(
                "w2_scale",
                torch.zeros(E, (H + BS - 1) // BS, nb, dtype=torch.float32),
            )

        elif qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            # 在线量化:buffer 先存源 dtype，process_weights 时再量化到 fp8。
            layer.w13 = nn.Parameter(
                torch.empty(E, 2 * M_tp, H, dtype=params_dtype), requires_grad=False
            )
            layer.w2 = nn.Parameter(
                torch.empty(E, H, M_tp, dtype=params_dtype), requires_grad=False
            )
            if qf == "fp8_per_tensor_online":
                layer.register_buffer("w13_scale", torch.zeros(E, dtype=torch.float32))
                layer.register_buffer("w2_scale", torch.zeros(E, dtype=torch.float32))
            elif qf == "fp8_per_channel_online":
                layer.register_buffer(
                    "w13_scale", torch.zeros(E, 2 * M_tp, dtype=torch.float32)
                )
                layer.register_buffer(
                    "w2_scale", torch.zeros(E, H, dtype=torch.float32)
                )
            else:  # fp8_per_block_online
                nb = (M_tp + BS - 1) // BS
                layer._n_scale_blocks_per_proj = nb
                layer.register_buffer(
                    "w13_scale",
                    torch.zeros(E, 2 * nb, (H + BS - 1) // BS, dtype=torch.float32),
                )
                layer.register_buffer(
                    "w2_scale",
                    torch.zeros(E, (H + BS - 1) // BS, nb, dtype=torch.float32),
                )
        else:
            raise ValueError(f"Fp8MoEMethod 不支持的 quant_family: {qf!r}")

    # ------------------------------------------------------------------ #
    #  dispatch_scale ← _dispatch_scale + 两个 copy 助手
    # ------------------------------------------------------------------ #
    def dispatch_scale(self, layer, local_id, proj, param_name, tensor):
        qf = layer._quant_family

        # online 家族不该收到已量化 ckpt 的 weight_scale，否则是 ckpt 与 QUANTIZATION 不匹配。
        if param_name in ("weight_scale", "weight_scale_inv") and qf in (
            "fp8_per_tensor_online",
            "fp8_per_channel_online",
            "fp8_per_block_online",
        ):
            raise RuntimeError(
                f"[Fp8MoEMethod] Got {param_name!r} for proj={proj!r} while in "
                f"online quant family {qf!r}; ckpt 已是预量化 FP8 但 QUANTIZATION 选了"
                f"在线路径。请改用 QUANTIZATION=FP8（已量化路径），或确保 ckpt 的"
                f" config.json 带 quantization_config 让 load_from_ckpt 自动选对。"
            )

        if param_name == "weight_scale" and qf == "fp8_per_tensor":
            scale_val = tensor.float().squeeze().item()
            if proj == "gate_proj":
                layer._gate_scales[local_id] = scale_val
            elif proj == "up_proj":
                layer._up_scales[local_id] = scale_val
            elif proj == "down_proj":
                layer._down_scales[local_id] = scale_val

        elif param_name == "weight_scale" and qf == "fp8_per_channel":
            self._copy_per_channel_scale(layer, local_id, proj, tensor)

        elif param_name == "weight_scale_inv" and qf == "fp8_per_block":
            self._copy_block_scale(layer, local_id, proj, tensor)

    def _copy_per_channel_scale(self, layer, expert_id, proj, tensor):
        scale = tensor.float().squeeze()
        if proj == "gate_proj":
            start = layer.tp_rank * layer.moe_inter_tp
            layer._gate_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "up_proj":
            start = layer.tp_rank * layer.moe_inter_tp
            layer._up_ch_scales.data[expert_id].copy_(
                scale.narrow(0, start, layer.moe_inter_tp)
            )
        elif proj == "down_proj":
            layer._down_ch_scales.data[expert_id].copy_(scale)

    def _copy_block_scale(self, layer, expert_id, proj, tensor):
        BS = layer._FP8_BLOCK_SIZE
        nb = layer._n_scale_blocks_per_proj
        if proj in ("gate_proj", "up_proj"):
            start_block = (layer.tp_rank * layer.moe_inter_tp) // BS
            sliced = tensor.narrow(0, start_block, nb).contiguous()
            row_start = nb if proj == "gate_proj" else 0
            layer.w13_scale.data[expert_id, row_start : row_start + nb].copy_(sliced)
        elif proj == "down_proj":
            start_block = (layer.tp_rank * layer.moe_inter_tp) // BS
            sliced = tensor.narrow(1, start_block, nb).contiguous()
            layer.w2_scale.data[expert_id].copy_(sliced)

    # ------------------------------------------------------------------ #
    #  process_weights_after_loading ← _fuse_* / _online_quantize_*
    # ------------------------------------------------------------------ #
    def process_weights_after_loading(self, layer):
        qf = layer._quant_family
        if qf == "fp8_per_tensor":
            self._fuse_per_tensor(layer)
        elif qf == "fp8_per_channel":
            self._fuse_per_channel(layer)
        elif qf == "fp8_per_tensor_online":
            self._online_per_tensor(layer)
        elif qf == "fp8_per_channel_online":
            self._online_per_channel(layer)
        elif qf == "fp8_per_block_online":
            self._online_per_block(layer)
        # fp8_per_block（已量化）无需后处理。

    def _fuse_per_tensor(self, layer):
        from rtp_llm.models_py.quant_methods.fp8 import _resolve_per_tensor_quant

        per_tensor_quant = _resolve_per_tensor_quant()
        M_tp = layer.moe_inter_tp
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(layer.w2.data, dtype=torch.float8_e4m3fn)

        for e in range(layer.num_local_experts):
            up_s = float(layer._up_scales[e].item())
            gate_s = float(layer._gate_scales[e].item())
            down_s = float(layer._down_scales[e].item())

            up_bf16 = layer.w13.data[e, :M_tp].float() * up_s
            gate_bf16 = layer.w13.data[e, M_tp:].float() * gate_s
            w13_bf16 = torch.cat([up_bf16, gate_bf16], dim=0).contiguous()

            qw13, sc13 = per_tensor_quant(w13_bf16)
            new_w13[e].copy_(qw13)
            layer.w13_scale[e] = sc13.view(-1)[0]

            w2_bf16 = layer.w2.data[e].float() * down_s
            qw2, sc2 = per_tensor_quant(w2_bf16.contiguous())
            new_w2[e].copy_(qw2)
            layer.w2_scale[e] = sc2.view(-1)[0]

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)
        del layer._gate_scales
        del layer._up_scales
        del layer._down_scales

    def _fuse_per_channel(self, layer):
        for e in range(layer.num_local_experts):
            layer.w13_scale.data[e, : layer.moe_inter_tp] = layer._up_ch_scales[e]
            layer.w13_scale.data[e, layer.moe_inter_tp :] = layer._gate_ch_scales[e]
            layer.w2_scale.data[e] = layer._down_ch_scales[e]
        del layer._gate_ch_scales
        del layer._up_ch_scales
        del layer._down_ch_scales

    def _online_per_tensor(self, layer):
        from rtp_llm.models_py.quant_methods.fp8 import _resolve_per_tensor_quant

        per_tensor_quant = _resolve_per_tensor_quant()
        E = layer.num_local_experts
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(layer.w2.data, dtype=torch.float8_e4m3fn)

        for e in range(E):
            qw, sc = per_tensor_quant(layer.w13.data[e].contiguous())
            new_w13[e].copy_(qw)
            layer.w13_scale[e] = sc.view(-1)[0]

            qw, sc = per_tensor_quant(layer.w2.data[e].contiguous())
            new_w2[e].copy_(qw)
            layer.w2_scale[e] = sc.view(-1)[0]

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_channel(self, layer):
        E = layer.num_local_experts
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(layer.w2.data, dtype=torch.float8_e4m3fn)
        fp8_max = _FP8_E4M3_MAX

        for e in range(E):
            w13_e = layer.w13.data[e].float()
            row_max = w13_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(_FP8_MIN_SCALE)
            new_w13[e] = (w13_e / row_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            layer.w13_scale.data[e] = row_scale

            w2_e = layer.w2.data[e].float()
            row_max = w2_e.abs().amax(dim=1)
            row_scale = (row_max / fp8_max).clamp_min(_FP8_MIN_SCALE)
            new_w2[e] = (w2_e / row_scale.unsqueeze(1)).to(torch.float8_e4m3fn)
            layer.w2_scale.data[e] = row_scale

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    def _online_per_block(self, layer):
        E = layer.num_local_experts
        BS = layer._FP8_BLOCK_SIZE
        device = layer.w13.data.device
        new_w13 = torch.empty_like(layer.w13.data, dtype=torch.float8_e4m3fn)
        new_w2 = torch.empty_like(layer.w2.data, dtype=torch.float8_e4m3fn)
        fp8_max = _FP8_E4M3_MAX

        rows13, cols13 = layer.w13.data.shape[1], layer.w13.data.shape[2]
        rows2, cols2 = layer.w2.data.shape[1], layer.w2.data.shape[2]
        nb_r13 = (rows13 + BS - 1) // BS
        nb_c13 = (cols13 + BS - 1) // BS
        nb_r2 = (rows2 + BS - 1) // BS
        nb_c2 = (cols2 + BS - 1) // BS

        for e in range(E):
            for bi in range(nb_r13):
                r0, r1 = bi * BS, min((bi + 1) * BS, rows13)
                for bj in range(nb_c13):
                    c0, c1 = bj * BS, min((bj + 1) * BS, cols13)
                    block = layer.w13.data[e, r0:r1, c0:c1].float()
                    scale = max(_FP8_MIN_SCALE, block.abs().max().item() / fp8_max)
                    new_w13[e, r0:r1, c0:c1] = (block / scale).to(torch.float8_e4m3fn)
                    layer.w13_scale.data[e, bi, bj] = scale
            for bi in range(nb_r2):
                r0, r1 = bi * BS, min((bi + 1) * BS, rows2)
                for bj in range(nb_c2):
                    c0, c1 = bj * BS, min((bj + 1) * BS, cols2)
                    block = layer.w2.data[e, r0:r1, c0:c1].float()
                    scale = max(_FP8_MIN_SCALE, block.abs().max().item() / fp8_max)
                    new_w2[e, r0:r1, c0:c1] = (block / scale).to(torch.float8_e4m3fn)
                    layer.w2_scale.data[e, bi, bj] = scale

        layer.w13 = nn.Parameter(new_w13.to(device), requires_grad=False)
        layer.w2 = nn.Parameter(new_w2.to(device), requires_grad=False)

    # ------------------------------------------------------------------ #
    #  add_weight_tensors ← _build_weights_dict 的 fp8 scale 注入
    # ------------------------------------------------------------------ #
    def add_weight_tensors(self, layer, weights_dict: Dict[str, Any]) -> None:
        weights_dict[W.moe_s1] = layer.w13_scale
        weights_dict[W.moe_s2] = layer.w2_scale
