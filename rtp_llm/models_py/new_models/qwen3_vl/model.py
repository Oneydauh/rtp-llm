"""Qwen3-VL dense (new-loader wrapper).

Combines the existing dense ``Qwen3ForCausalLM`` LLM (qk-norm + tied lm_head,
already correct for Qwen3-VL's text tower) with a Qwen3-VL DeepStack vision
tower. The ckpt lays weights out as::

    model.language_model.*   -> language_model.*
    model.visual.*           -> visual.*
    (no lm_head; tie_word_embeddings=True)

The top-level mapper rewrites those prefixes; ``_groupby_prefix`` then dispatches
each subtree to the matching child's ``load_weights``. ``Qwen3ForCausalLM`` ties
lm_head to embed_tokens itself when no ``lm_head.weight`` is present.
"""

from typing import Any

from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3_vl.vision import Qwen3VLVisionTransformer
from rtp_llm.models_py.weight_mapper import WeightsMapper


class Qwen3VLForConditionalGeneration(RtpModule):

    # Longest-prefix-first ordering is handled inside WeightsMapper, so
    # "model.visual." / "model.language_model." win over any shorter key.
    WEIGHTS_MAPPER = WeightsMapper(
        prefix_mapping={
            "model.visual.": "visual.",
            "model.language_model.": "language_model.",
            "visual.": "visual.",
            "lm_head.": "language_model.lm_head.",
        }
    )

    def __init__(self, model_config: Any, load_config: Any):
        super().__init__()
        self.model_config = model_config
        self.load_config = load_config
        vit_config = self._get_vit_config(model_config)

        self.visual = Qwen3VLVisionTransformer(
            vit_config=vit_config, load_config=load_config
        )
        self.language_model = Qwen3ForCausalLM(
            model_config=model_config, load_config=load_config
        )

    def initialize(self, init_resource) -> bool:
        return self.language_model.initialize(init_resource)

    def prepare_fmha_impl(self, inputs, is_cuda_graph: bool = False):
        return self.language_model.prepare_fmha_impl(inputs, is_cuda_graph)

    def load_weights(self, weights):
        if isinstance(weights, dict):
            weights_iter = iter(weights.items())
        else:
            weights_iter = weights

        mapped_iter = self.WEIGHTS_MAPPER.apply(weights_iter)
        grouped = self._groupby_prefix(mapped_iter)

        for prefix, sub_weights in grouped.items():
            child = self._get_child_module(prefix)
            if child is not None and hasattr(child, "load_weights"):
                child.load_weights(sub_weights)

    def _get_vit_config(self, model_config) -> dict:
        # Old-side _create_config stores the HF vision_config dict on
        # mm_related_params.config; fall back to a direct vision_config attr or
        # the model defaults of Qwen3-VL-4B.
        mm = getattr(model_config, "mm_related_params", None)
        if mm is not None and getattr(mm, "config", None):
            return mm.config
        if hasattr(model_config, "vision_config"):
            return model_config.vision_config
        if isinstance(model_config, dict):
            return model_config.get("vision_config", {})
        return {
            "hidden_size": 1024,
            "num_heads": 16,
            "depth": 24,
            "intermediate_size": 4096,
            "patch_size": 16,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "spatial_merge_size": 2,
            "out_hidden_size": 2560,
            "num_position_embeddings": 2304,
            "deepstack_visual_indexes": [5, 11, 17],
        }

    def forward(self, inputs, fmha_impl: Any = None):
        return self.language_model.forward(inputs, fmha_impl=fmha_impl)
