"""Qwen3-VL dense — config / architecture registration.

This class only provides ckpt-config parsing and the
``Qwen3VLForConditionalGeneration`` architecture registration so the model is
selectable and a ``ModelConfig`` can be built. The actual weight loading + LLM
forward go through the NEW loader (``rtp_llm/models_py/new_models/qwen3_vl``),
which is selected when ``USE_NEW_LOADER=1`` (or model_config.use_new_loader).

Qwen3-VL nests the language hyper-params under ``text_config`` (unlike
Qwen2-VL which keeps them top-level), so ``_from_hf`` reads from there. Vision
params + multimodal token ids stay top-level and reuse QWen2_VL._load_vit_param.
"""

import json
import os
from typing import Any, Dict

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.model_factory_register import register_model
from rtp_llm.models.qwen2_vl.qwen2_vl import QWen2_VL


class QWen3_VL(QWen2_VL):
    def _init_multimodal_for_new_loader(self):
        """New-loader hook (called from base_model after py_model is built).

        Vision weights already live in self.py_model.visual, so just run the
        normal multimodal init — _init_multimodal below reuses that tower
        instead of loading a fresh one.
        """
        self._may_init_multimodal()

    def _init_multimodal(self, mm_model_config, vit_config):
        from rtp_llm.models.qwen3_vl.qwen3_vl_image_embedding import (
            Qwen3VLImageEmbedding,
        )

        visual = getattr(getattr(self, "py_model", None), "visual", None)
        assert (
            visual is not None
        ), "QWen3_VL multimodal expects new-loader py_model.visual to exist"
        self.mm_part = Qwen3VLImageEmbedding(
            self.model_config.mm_related_params,
            visual,
            model_config=self.model_config,
        )
        # No vit_weights needed: under the new loader the vision weights are
        # already loaded into py_model.visual; the old load_mm_weight path
        # (which consumes vit_weights) is skipped.

    @classmethod
    def _create_config(cls, ckpt_path: str) -> ModelConfig:
        config = ModelConfig()
        config.ckpt_path = ckpt_path

        config_path = os.path.join(ckpt_path, "config.json")
        if not os.path.exists(config_path):
            return config
        with open(config_path) as reader:
            config_json = json.loads(reader.read())

        cls._from_hf(config, config_json)
        # vision_config + vision_start/end token ids live at top level.
        QWen2_VL._load_vit_param(config, config_json)
        config.mm_related_params.config["ckpt_path"] = ckpt_path
        return config

    @staticmethod
    def _from_hf(config: ModelConfig, config_json: Dict[str, Any]):
        # Language hyper-params are nested under text_config in Qwen3-VL.
        tc = config_json.get("text_config", config_json)

        config.vocab_size = tc["vocab_size"]
        config.max_seq_len = tc.get("max_position_embeddings", 10240)
        config.activation_type = "SiGLU"
        config.hidden_size = tc["hidden_size"]
        config.attn_config.head_num = tc["num_attention_heads"]
        config.attn_config.kv_head_num = tc["num_key_value_heads"]
        config.attn_config.size_per_head = (
            int(tc["head_dim"])
            if "head_dim" in tc
            else tc["hidden_size"] // tc["num_attention_heads"]
        )
        config.num_layers = tc["num_hidden_layers"]
        config.inter_size = tc["intermediate_size"]
        config.norm_type = "rmsnorm"
        config.layernorm_eps = tc["rms_norm_eps"]
        config.has_post_decoder_layernorm = True
        # Qwen3 applies per-head RMSNorm on Q/K (the new-loader module does this
        # unconditionally; flag kept for parity with the old qwen_3 path).
        config.qk_norm = True

        config.special_tokens.bos_token_id = tc.get(
            "bos_token_id", config_json.get("bos_token_id", -1)
        )
        config.special_tokens.eos_token_id = tc.get(
            "eos_token_id", config_json.get("eos_token_id", 0)
        )
        config.tie_word_embeddings = tc.get(
            "tie_word_embeddings", config_json.get("tie_word_embeddings", False)
        )

        # M-RoPE is NOT yet plumbed on the *new-loader py fmha* path: the 3D
        # (t/h/w) combo_position_ids the Mrope kernel needs are not fed into the
        # py model's attention (only the old-loader C++ forward path plumbs
        # them). Setting RopeStyle::Mrope (7) there makes the rope kernel read
        # invalid positions -> CUBLAS_STATUS_EXECUTION_FAILED at the next GEMM.
        # Until step 2 plumbs 3D position ids into the py path, use plain 1D
        # rope: text positions are exact, image-token positions are approximate.
        config.mm_model_config.mm_position_ids_style = 0
        rope_config = config.attn_config.rope_config
        rope_config.style = 1
        rope_config.base = int(tc["rope_theta"])
        rope_config.dim = int(config.attn_config.size_per_head)


register_model("qwen3_vl", QWen3_VL, ["Qwen3VLForConditionalGeneration"])
