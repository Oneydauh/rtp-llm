"""
DeepSeek V3 MTP (Multi-Token Prediction) draft head for new-loader.

Top-level model: DeepSeekV32MTPForCausalLM
  - HF ckpt has a single-layer structure under "model.layers.0."
  - WEIGHTS_MAPPER strips "model.layers.0." prefix.
  - load_weights() remaps the stripped keys to match submodule names:
      embed_tokens.weight        -> embed_tokens.weight
      shared_head.head.weight    -> lm_head.weight
      shared_head.norm.weight    -> norm.weight
      enorm.weight               -> mtp_block.e_norm.weight
      hnorm.weight               -> mtp_block.h_norm.weight
      eh_proj.weight             -> mtp_block.fc.weight  (transposed!)
      self_attn.*                -> layers.0.self_attn.*
      mlp.*                      -> layers.0.mlp.*
      input_layernorm.*          -> layers.0.input_layernorm.*
      post_attention_layernorm.* -> layers.0.post_attention_layernorm.*
"""

import json
import logging
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.block_map import select_block_map_for_layer
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.new_models.deepseek_v3.language import (
    MlaKernelWeightLayout,
    _build_rope_cache,
    _extract_config_values,
    _read_config_json,
)
from rtp_llm.models_py.new_models.deepseek_v3.model import DeepSeekV32DecoderLayer
from rtp_llm.models_py.new_models.mtp import MTPBlock
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Key remapping helpers
# ------------------------------------------------------------------ #

# After WEIGHTS_MAPPER strips "model.layers.0.", these are the remaining
# key prefixes / full names that need to be renamed to match submodule paths.
_LAYER_PREFIXES = (
    "self_attn.",
    "mlp.",
    "input_layernorm.",
    "post_attention_layernorm.",
)

_KEY_REMAP = {
    "shared_head.head.weight": "lm_head.weight",
    "shared_head.norm.weight": "norm.weight",
    "enorm.weight": "mtp_block.e_norm.weight",
    "hnorm.weight": "mtp_block.h_norm.weight",
}

_EH_PROJ_KEY = "eh_proj.weight"
_EH_PROJ_MAPPED = "mtp_block.fc.weight"


def _remap_key(name: str):
    """Remap a stripped HF key to the submodule key used in this model.

    Returns (new_name, needs_transpose).
    """
    # eh_proj needs transpose
    if name == _EH_PROJ_KEY:
        return _EH_PROJ_MAPPED, True

    # simple renames
    if name in _KEY_REMAP:
        return _KEY_REMAP[name], False

    # layer-level weights: prepend "layers.0."
    for prefix in _LAYER_PREFIXES:
        if name.startswith(prefix):
            return "layers.0." + name, False

    # embed_tokens.weight and anything else: pass through unchanged
    return name, False


# ------------------------------------------------------------------ #
#  Top-level model
# ------------------------------------------------------------------ #


class DeepSeekV32MTPForCausalLM(GptModelBase):
    """DeepSeek V3 MTP draft head for new-loader.

    Single-layer MoE decoder + MTPBlock projection.
    HF ckpt prefix "model.layers.0." is stripped by WEIGHTS_MAPPER.
    """

    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.layers.0.": ""})

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        def _mapped(it):
            for name, tensor in it:
                new_name, needs_transpose = _remap_key(name)
                if needs_transpose:
                    tensor = tensor.t().contiguous()
                yield new_name, tensor

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        super().load_weights(_mapped(mapped_iter))

    def __init__(
        self,
        model_config: Any,
        load_config: Any,
    ):
        parallelism_config = getattr(load_config, "parallelism_config", None)
        fmha_config = getattr(load_config, "fmha_config", None)
        device_resource_config = getattr(load_config, "device_resource_config", None)

        super().__init__(
            config=model_config,
            parallelism_config=parallelism_config,
            weight=None,
            max_generate_batch_size=0,
            fmha_config=fmha_config,
            device_resource_config=device_resource_config,
        )

        # Resolve ckpt_path
        ckpt_path = ""
        if hasattr(model_config, "ckpt_path") and model_config.ckpt_path:
            ckpt_path = model_config.ckpt_path

        config_json = _read_config_json(ckpt_path)
        cfg = _extract_config_values(model_config, load_config, config_json)

        # MTP model always has exactly 1 layer and it is a MoE layer
        # Override moe_layer_index to ensure layer 0 is treated as MoE
        cfg["moe_layer_index"] = [0]
        cfg["num_layers"] = 1

        # --- RoPE cache ---
        device = torch.device("cuda")
        cos_sin_cache = _build_rope_cache(
            config_json if config_json else cfg,
            cfg["max_seq_len"],
            device,
        )
        self.register_buffer("cos_sin_cache", cos_sin_cache, persistent=False)

        # --- Embedding ---
        self.embed_tokens = VocabParallelEmbedding(
            vocab_size=cfg["vocab_size"],
            embedding_dim=cfg["hidden_size"],
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            params_dtype=cfg["params_dtype"],
        )

        # --- MTP projection block (reverse_concat=True for DeepSeek style) ---
        # HF ckpt names: enorm.weight -> e_norm, hnorm.weight -> h_norm,
        # eh_proj.weight -> fc  (all mapped in load_weights above)
        self.mtp_block = MTPBlock(
            hidden_size=cfg["hidden_size"],
            rms_norm_eps=cfg["rms_norm_eps"],
            reverse_concat=True,
            bias=False,
            params_dtype=cfg["params_dtype"],
        )

        # --- Single MoE decoder layer ---
        self.layers = nn.ModuleList()
        layer = DeepSeekV32DecoderLayer(
            hidden_size=cfg["hidden_size"],
            num_heads=cfg["num_heads"],
            q_lora_rank=cfg["q_lora_rank"],
            kv_lora_rank=cfg["kv_lora_rank"],
            nope_head_dim=cfg["nope_head_dim"],
            rope_head_dim=cfg["rope_head_dim"],
            v_head_dim=cfg["v_head_dim"],
            layer_idx=0,
            tp_size=cfg["tp_size"],
            tp_rank=cfg["tp_rank"],
            ep_size=cfg["ep_size"],
            ep_rank=cfg["ep_rank"],
            params_dtype=cfg["params_dtype"],
            layernorm_eps=cfg["rms_norm_eps"],
            quant_config=cfg["quant_config"],
            model_config=cfg["model_config"],
            parallelism_config=cfg["parallelism_config"],
            moe_config=cfg["moe_config"],
            is_moe_layer=True,
            dense_intermediate_size=cfg["dense_intermediate_size"],
            moe_intermediate_size=cfg["moe_intermediate_size"],
            num_experts=cfg["num_experts"],
            top_k=cfg["top_k"],
            shared_expert_intermediate_size=cfg["shared_expert_intermediate_size"],
            has_shared_expert=True,
            scoring_func=cfg["scoring_func"],
            routed_scaling_factor=cfg["routed_scaling_factor"],
            n_group=cfg["n_group"],
            topk_group=cfg["topk_group"],
            has_moe_norm=cfg["has_moe_norm"],
            correction_bias=cfg["has_e_score_correction"],
            is_sparse=cfg["is_sparse"],
            index_n_heads=cfg["indexer_head_num"],
            index_head_dim=cfg["indexer_head_dim"],
            index_topk=cfg["indexer_topk"],
            indexer_is_neox_style=cfg["indexer_is_neox_style"],
            cos_sin_cache=cos_sin_cache,
            blocksize=cfg["blocksize"],
        )
        self.layers.append(layer)

        # --- Final norm (from shared_head.norm) ---
        self.norm = RMSResNorm(
            cfg["hidden_size"],
            eps=cfg["rms_norm_eps"],
            params_dtype=cfg["params_dtype"],
        )

        # --- LM head (from shared_head.head) ---
        self.lm_head = ParallelLMHead(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            tp_size=cfg["lm_head_tp_size"],
            tp_rank=cfg["lm_head_tp_rank"],
            params_dtype=cfg["lm_head_params_dtype"],
        )

    def initialize(self, init_resource):
        """Build the MLA kernel layout after all post-load hooks have run."""
        ok = super().initialize(init_resource)
        self._ensure_mla_kernel_layout()
        return ok

    def _ensure_mla_kernel_layout(self) -> None:
        """Reconstruct only the W.* tensor views required by MLA kernels."""
        if getattr(self, "_mla_kernel_layout", None) is not None:
            return
        self._mla_kernel_layout = MlaKernelWeightLayout(
            [layer.self_attn._build_mla_kernel_weights() for layer in self.layers],
            self.cos_sin_cache,
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> Any:
        self._ensure_mla_kernel_layout()
        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self._mla_kernel_layout,
            inputs.attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        # inputs_embeds from current token ids
        inputs_embeds = self.embed_tokens(inputs.input_ids)
        # MTP block: combine embed with last hidden states from the main model
        hidden_states = self.mtp_block(inputs_embeds, inputs.input_hiddens)
        residual = torch.zeros_like(hidden_states)

        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)

        for i, layer in enumerate(self.layers):
            select_block_map_for_layer(inputs.attention_inputs, i)
            hidden_states, residual = layer(
                hidden_states,
                residual,
                fmha_impl,
                kv_cache=self.kv_cache.get_layer_cache(i) if self.kv_cache else None,
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return PyModelOutputs(hidden_states, fmha_impl.fmha_params)
