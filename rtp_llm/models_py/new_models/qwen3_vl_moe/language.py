"""Qwen3-VL-MoE language model for the new loader.

语言骨干与 Qwen3-MoE 完全一致(旧 loader 里 ``QWen3_VL_MOE.get_weight_cls()`` 是
``QWen3VLMoeWeightInfo``,继承 ``QWenV3MoeWeight``),因此复用 ``new_models/qwen3_moe`` 的
``Qwen3MoeForCausalLM``;只在 forward 上加多模态 embedding 注入 + 逐层 deepstack 注入
(与 ``model_desc/qwen3vl_moe.py`` 对齐)。结构与 ``new_models/qwen3_vl`` 完全平行,
区别只是基类从稠密换成 MoE。
"""

import json
import os
from pathlib import Path
from typing import Any

import torch

from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.modules import (
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    reshape_extra_input_to_deepstack,
)
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs


def _qwen3vl_moe_dump_tensor(
    tag: str, tensor: torch.Tensor, layer_idx: int = -1
) -> None:
    dump_dir = os.environ.get("QWEN3_VL_MOE_TENSOR_DUMP_DIR")
    if not dump_dir or not isinstance(tensor, torch.Tensor):
        return
    try:
        rank = (
            os.environ.get("WORLD_RANK")
            or os.environ.get("RANK")
            or os.environ.get("LOCAL_RANK")
            or "0"
        )
        path = Path(dump_dir)
        path.mkdir(parents=True, exist_ok=True)
        x = tensor.detach()
        xf = x.float()
        flat = xf.reshape(-1)
        sample = flat[:8].cpu().tolist()
        row = {
            "tag": tag,
            "layer": layer_idx,
            "rank": rank,
            "pid": os.getpid(),
            "shape": list(x.shape),
            "dtype": str(x.dtype),
            "mean": float(xf.mean().item()) if flat.numel() else 0.0,
            "std": float(xf.std(unbiased=False).item()) if flat.numel() else 0.0,
            "absmax": float(xf.abs().max().item()) if flat.numel() else 0.0,
            "sum": float(xf.sum().item()) if flat.numel() else 0.0,
            "l2": float(torch.linalg.vector_norm(xf).item()) if flat.numel() else 0.0,
            "sample": sample,
        }
        with (path / f"rank{rank}_pid{os.getpid()}.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    except Exception as exc:
        logging.warning("qwen3vl_moe tensor dump failed for %s: %s", tag, exc)


class Qwen3VLMoeForCausalLM(Qwen3MoeForCausalLM):

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__(model_config, load_config)
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids = inputs.input_ids
        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask

        mm = inputs.multimodal_inputs
        mm_features = mm.multimodal_features
        mm_feature_locs = mm.mm_features_locs
        mm_extra_input = mm.mm_extra_input
        mm_deepstack_embeds = (
            reshape_extra_input_to_deepstack(mm_extra_input, mm_features)
            if mm_extra_input
            else []
        )

        do_dump = bool(os.environ.get("QWEN3_VL_MOE_TENSOR_DUMP_DIR")) and not getattr(
            self, "_qwen3vl_moe_dumped", False
        )
        if do_dump:
            self._qwen3vl_moe_dumped = True

        inputs_embeds = self.embed_tokens(
            input_ids, position_ids, token_type_ids, text_tokens_mask
        )
        if do_dump:
            _qwen3vl_moe_dump_tensor("mm_features", mm_features)
            _qwen3vl_moe_dump_tensor("mm_feature_locs", mm_feature_locs)
            _qwen3vl_moe_dump_tensor("mm_extra_input", mm_extra_input)
            for _ds_idx, _ds in enumerate(mm_deepstack_embeds[:3]):
                _qwen3vl_moe_dump_tensor(f"deepstack_{_ds_idx}", _ds)
            _qwen3vl_moe_dump_tensor("embed", inputs_embeds)
        hidden_states = self.multimodal_embedding_injector(
            inputs_embeds, mm_features, mm_feature_locs
        )
        if do_dump:
            _qwen3vl_moe_dump_tensor("after_mm_embed", hidden_states)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        if mm_deepstack_embeds and mm_feature_locs is not None:
            cpu_locs = (
                mm_feature_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            )
        else:
            cpu_locs = []

        residual = torch.zeros_like(hidden_states)
        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            if do_dump:
                _qwen3vl_moe_dump_tensor("layer_out", hidden_states, i)
                _qwen3vl_moe_dump_tensor("layer_residual", residual, i)
            hidden_states = self.multimodal_deepstack_injector(
                hidden_states, mm_deepstack_embeds, cpu_locs, i
            )
            if do_dump:
                _qwen3vl_moe_dump_tensor("after_deepstack", hidden_states, i)

        hidden_states, _ = self.norm(hidden_states, residual)
        if do_dump:
            _qwen3vl_moe_dump_tensor("final_norm", hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLMoeForCausalLM"]
