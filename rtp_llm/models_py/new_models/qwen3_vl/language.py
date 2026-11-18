"""Qwen3-VL text owners for main's multimodal serving interface."""

import torch

from rtp_llm.models_py.new_models.model_base import select_fmha_impl_for_layer
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3_moe.language import Qwen3MoeForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.multimodal import Qwen3VLMultimodalMixin
from rtp_llm.models_py.weight_mapper import WeightsMapper
from rtp_llm.ops.compute_ops import PyModelOutputs


class _LanguageRuntime(Qwen3VLMultimodalMixin):
    # Visual parameters are loaded by the service's independent vision tower.
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.language_model.": ""})
    _is_moe = False

    def __init__(self, model_config, load_config):
        super().__init__(model_config, load_config)
        self._init_multimodal_injectors()

    def checkpoint_weight_name_filter(self):
        owns_stage = super().checkpoint_weight_name_filter()
        return lambda key: (
            key.startswith("model.language_model.") or key.startswith("lm_head.")
        ) and (owns_stage is None or owns_stage(key))

    def forward(self, inputs, fmha_impl=None):
        hidden, deepstack, locations = self._embed_multimodal_inputs(inputs)
        residual = torch.zeros_like(hidden) if self._is_moe else None
        if fmha_impl is None:
            fmha_impl = self.prepare_fmha_impl(inputs)
        capture = self.capture_context(inputs.capture_hidden_states)
        layers = self.layers.values() if self._pipeline_partitioned else self.layers
        for local, (layer_id, layer) in enumerate(zip(self._owned_layer_ids, layers)):
            attention = select_fmha_impl_for_layer(fmha_impl, self.kv_cache, local)
            cache = self.kv_cache.get_layer_cache(local) if self.kv_cache else None
            if self._is_moe:
                hidden, residual = layer(hidden, residual, attention, cache)
            else:
                hidden = layer(hidden, attention, kv_cache=cache)
            hidden = self._inject_deepstack_after_layer(
                hidden, deepstack, locations, layer_id
            )
            if self._is_moe:
                capture.capture_layer(layer_id, hidden, residual)
            else:
                capture.capture_layer(layer_id, hidden)
        if self.norm is None:
            intermediate = hidden + residual if self._is_moe else hidden
            out = PyModelOutputs(intermediate)
            out.pp_intermediates = {"hidden_states": intermediate}
            return out
        return (
            capture.finalize(hidden, residual)
            if self._is_moe
            else capture.finalize(hidden)
        )


class Qwen3VLForCausalLM(_LanguageRuntime, Qwen3ForCausalLM):
    pass


class Qwen3VLMoeForCausalLM(_LanguageRuntime, Qwen3MoeForCausalLM):
    _is_moe = True
