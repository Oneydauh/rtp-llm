"""Main runtime entry for GLM4 non-Lite (GQA), not the MLA model."""

import math

import torch
from torch import nn

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.models_py.layers.embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
    output_vocab_head_kwargs,
)
from rtp_llm.models_py.layers.norm import RMSResNorm
from rtp_llm.models_py.model_desc.module_base import GptModelBase
from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.qwen3.language import (
    Qwen3ForCausalLM,
    _extract_config_values,
    _validate_supported_parallelism,
)
from rtp_llm.ops.compute_ops import PyModelOutputs

from .checkpoint_layout import Glm4CheckpointLayout
from .decoder import Glm4MoeDecoderLayer


class Glm4MoeForCausalLM(Qwen3ForCausalLM):
    """Only the common embedding/tied-head checkpoint lifecycle is inherited."""

    # Expert/projection updates need explicit owner preflight, not the dense
    # model's blanket checkpoint-update capability.
    supports_checkpoint_weight_updates = False

    def validate_lora_checkpoint_target(self, name):
        from rtp_llm.models_py.lora_checkpoint import (
            validate_attention_lora_checkpoint_target,
        )

        validate_attention_lora_checkpoint_target(name, "GLM4 MoE")

    def __init__(self, model_config, load_config):
        if not isinstance(model_config, ModelConfig):
            raise TypeError("GLM4 MoE requires a typed ModelConfig")
        parallel = load_config.parallelism_config
        _validate_supported_parallelism(parallel)
        GptModelBase.__init__(
            self,
            config=model_config,
            parallelism_config=parallel,
            weight=None,
            max_generate_batch_size=load_config.max_generate_batch_size,
            fmha_config=load_config.fmha_config,
            py_hw_kernel_config=load_config.py_hw_kernel_config,
        )
        cfg = _extract_config_values(
            model_config, load_config, validate_intermediate_size=False
        )
        cfg.update(
            num_experts=model_config.expert_num,
            top_k=model_config.moe_k,
            moe_intermediate_size=model_config.moe_inter_size,
            model_config=model_config,
            parallelism_config=parallel,
            moe_config=load_config.moe_config,
            ep_size=load_config.ep_size,
            ep_rank=load_config.ep_rank,
        )
        self._layout = Glm4CheckpointLayout.read(model_config)
        self._owned_layer_ids = self.layer_ids()
        self._pipeline_partitioned = getattr(parallel, "pp_size", 1) > 1
        first = 0 in self._owned_layer_ids
        last = cfg["num_layers"] - 1 in self._owned_layer_ids
        self.tie_word_embeddings = model_config.tie_word_embeddings
        self.normalize_lm_head_weight = model_config.normalize_lm_head_weight
        self.logit_scale = float(model_config.logit_scale)
        if not math.isfinite(self.logit_scale):
            raise ValueError("logit_scale must be finite")
        self._lm_head_postprocessed = False
        self.embed_tokens = (
            VocabParallelEmbedding(
                cfg["vocab_size"],
                cfg["hidden_size"],
                tp_size=cfg["attn_tp_size"],
                tp_rank=cfg["attn_tp_rank"],
                params_dtype=cfg["params_dtype"],
                allow_vocab_padding=True,
            )
            if first
            else None
        )
        owned = {
            str(i): Glm4MoeDecoderLayer(
                cfg,
                i,
                dense_width=self._layout.dense_width,
                shared_width=self._layout.shared_width,
                moe_layers=self._layout.routed_layers,
                use_qk_norm=self._layout.use_qk_norm,
                correction_bias=self._layout.correction_bias,
            )
            for i in self._owned_layer_ids
        }
        self.layers = (
            nn.ModuleDict(owned)
            if self._pipeline_partitioned
            else nn.ModuleList(owned.values())
        )
        self.norm = (
            RMSResNorm(
                cfg["hidden_size"],
                eps=cfg["rms_norm_eps"],
                params_dtype=cfg["params_dtype"],
            )
            if last
            else None
        )
        self.lm_head = (
            ParallelLMHead(
                cfg["vocab_size"],
                cfg["hidden_size"],
                tp_size=cfg["lm_head_tp_size"],
                tp_rank=cfg["lm_head_tp_rank"],
                params_dtype=cfg["lm_head_params_dtype"],
                **output_vocab_head_kwargs(model_config)
            )
            if last
            else None
        )
        if self.tie_word_embeddings and first and last:
            self.embed_tokens.tie_output_head(self.lm_head)
        self._init_capture_context(
            self._capture_canonical_layer, self._capture_canonical_final
        )

    def checkpoint_weight_name_filter(self):
        stage_owns = super().checkpoint_weight_name_filter()
        if stage_owns is None:
            return self._layout.target_owns
        return lambda name: self._layout.target_owns(name) and stage_owns(name)

    def _capture_canonical_layer(self, hidden_states, residual):
        return hidden_states + residual

    def _capture_canonical_final(self, hidden_states, residual):
        return self.norm(hidden_states, residual)[0]

    def forward(self, inputs, fmha_impl=None):
        hidden = (
            self.embed_tokens(inputs.input_ids)
            if self.embed_tokens is not None
            else inputs.pp_intermediates["hidden_states"]
        )
        residual = torch.zeros_like(hidden)
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)
        layers = self.layers.values() if self._pipeline_partitioned else self.layers
        for local, (global_id, layer) in enumerate(zip(self._owned_layer_ids, layers)):
            hidden, residual = layer(
                hidden,
                residual,
                select_fmha_impl_for_layer(fmha_impl, self.kv_cache, local),
                (
                    self.kv_cache.get_layer_cache(local)
                    if self.kv_cache is not None
                    else None
                ),
            )
            capture.capture_layer(global_id, hidden, residual)
        if self.norm is None:
            materialized = hidden + residual
            out = PyModelOutputs(materialized)
            out.pp_intermediates = {"hidden_states": materialized}
            return out
        return capture.finalize(hidden, residual)
