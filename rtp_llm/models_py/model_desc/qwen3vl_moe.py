import json
import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Type

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.generic_moe import GenericMoeDecoderLayer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import (
    AttnImplFactory,
    Embedding,
    MultimodalDeepstackInjector,
    MultimodalEmbeddingInjector,
    RMSResNorm,
    reshape_extra_input_to_deepstack,
)
from rtp_llm.ops import MoeConfig, ParallelismConfig
from rtp_llm.ops.compute_ops import PyAttentionInputs, PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W


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


class Qwen3VLMoeModel(GptModelBase):
    """Qwen3VL MoE model"""

    def __init__(
        self,
        model_config: ModelConfig,
        parallelism_config: ParallelismConfig,
        weights: ModelWeights,
        moe_config: MoeConfig,
        max_generate_batch_size: int,
        fmha_config=None,
        py_hw_kernel_config=None,
        device_resource_config=None,
    ):
        super().__init__(
            model_config,
            parallelism_config,
            weights,
            max_generate_batch_size=max_generate_batch_size,
            fmha_config=fmha_config,
            py_hw_kernel_config=py_hw_kernel_config,
            device_resource_config=device_resource_config,
        )
        # Determine attention_type from model_config.attn_config.use_mla
        self.embed_tokens = Embedding(
            model_config, parallelism_config, weights.get_global_weight(W.embedding)
        )
        self.multimodal_embedding_injector = MultimodalEmbeddingInjector()
        self.multimodal_deepstack_injector = MultimodalDeepstackInjector()
        # Get enable_cuda_graph from py_hw_kernel_config
        enable_cuda_graph = (
            py_hw_kernel_config.enable_cuda_graph
            if py_hw_kernel_config is not None
            else False
        )
        self.layers = nn.ModuleList(
            [
                GenericMoeDecoderLayer(
                    model_config,
                    parallelism_config,
                    weights.weights[idx],
                    weights.global_weights,
                    idx,
                    moe_config,
                    max_generate_batch_size,
                    hw_kernel_config=py_hw_kernel_config,
                    enable_cuda_graph=enable_cuda_graph,
                )
                for idx in range(self.layer_num)
            ]
        )
        self.norm = RMSResNorm(
            weights.get_global_weight(W.final_ln_gamma), eps=model_config.layernorm_eps
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        input_ids: torch.Tensor = inputs.input_ids

        position_ids = inputs.combo_position_ids
        token_type_ids = inputs.embedding_inputs.combo_tokens_type_ids
        text_tokens_mask = inputs.embedding_inputs.text_tokens_mask
        mm_features = inputs.multimodal_inputs.multimodal_features
        mm_feature_locs = inputs.multimodal_inputs.mm_features_locs
        mm_extra_input = inputs.multimodal_inputs.mm_extra_input
        # extra input arrives as flat 1-D tensors; reshape back to deepstack [layers, tokens, hidden]
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

        if mm_deepstack_embeds and mm_feature_locs is not None:
            cpu_locs = (
                mm_feature_locs.to(device="cpu", dtype=torch.long).view(-1).tolist()
            )
        else:
            cpu_locs = []

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        residual = torch.zeros_like(hidden_states)
        for i, decoder_layer in enumerate(self.layers[: self.layer_num]):
            select_block_map_for_layer(inputs.attention_inputs, i)
            output = decoder_layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )
            if do_dump:
                _qwen3vl_moe_dump_tensor("layer_out", output.hidden_states, i)
                _qwen3vl_moe_dump_tensor("layer_residual", output.residual, i)
            hidden_states = self.multimodal_deepstack_injector(
                output.hidden_states, mm_deepstack_embeds, cpu_locs, i
            )
            if do_dump:
                _qwen3vl_moe_dump_tensor("after_deepstack", hidden_states, i)
            residual = output.residual

        hidden_states, _ = self.norm(hidden_states, residual)
        if do_dump:
            _qwen3vl_moe_dump_tensor("final_norm", hidden_states)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)


__all__ = ["Qwen3VLMoeModel"]
