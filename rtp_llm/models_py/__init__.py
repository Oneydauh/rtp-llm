import logging

from rtp_llm.models_py.model_loader import LoadConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule, rtp_module
from rtp_llm.models_py.registry import MODEL_REGISTRY, get_model_class, register_model

_logger = logging.getLogger(__name__)

# Map exported class/alias names to their implementation modules and class names.
_EXPORTED_MODELS = {
    "DeepSeekV32ForCausalLM": (
        "rtp_llm.models_py.new_models.deepseek_v3",
        "DeepSeekV32ForCausalLM",
    ),
    "glm_5": ("rtp_llm.models_py.new_models.deepseek_v3", "DeepSeekV32ForCausalLM"),
    "Qwen2VLForConditionalGeneration": (
        "rtp_llm.models_py.new_models.qwen2_vl",
        "Qwen2VLForConditionalGeneration",
    ),
    "Qwen2ForCausalLM": (
        "rtp_llm.models_py.new_models.qwen2_vl.language",
        "Qwen2ForCausalLM",
    ),
    "Qwen3ForCausalLM": ("rtp_llm.models_py.new_models.qwen3", "Qwen3ForCausalLM"),
    "Qwen3MoeForCausalLM": (
        "rtp_llm.models_py.new_models.qwen3_moe",
        "Qwen3MoeForCausalLM",
    ),
    "Qwen3VLForConditionalGeneration": (
        "rtp_llm.models_py.new_models.qwen3_vl",
        "Qwen3VLForConditionalGeneration",
    ),
    "Qwen3VLMoeForConditionalGeneration": (
        "rtp_llm.models_py.new_models.qwen3_vl_moe",
        "Qwen3VLMoeForConditionalGeneration",
    ),
    "Glm4MoeForCausalLM": ("rtp_llm.models_py.new_models.glm", "Glm4MoeForCausalLM"),
    "ChatGLMForCausalLM": ("rtp_llm.models_py.new_models.glm", "ChatGLMForCausalLM"),
    "Qwen2MTPForCausalLM": (
        "rtp_llm.models_py.new_models.qwen2_mtp",
        "Qwen2MTPForCausalLM",
    ),
    "DeepSeekV32MTPForCausalLM": (
        "rtp_llm.models_py.new_models.deepseek_v3_mtp",
        "DeepSeekV32MTPForCausalLM",
    ),
}


def __getattr__(name: str):
    if name in _EXPORTED_MODELS:
        import importlib

        module_path, class_name = _EXPORTED_MODELS[name]
        try:
            mod = importlib.import_module(module_path)
            return getattr(mod, class_name)
        except Exception as e:
            _logger.warning(
                "Failed to dynamic-import '%s' (from %s.%s): %s",
                name,
                module_path,
                class_name,
                e,
            )
            raise ImportError(
                f"Failed to dynamically import {name} from {module_path}: {e}"
            ) from e
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "get_model_class",
    "NewModelLoader",
    "LoadConfig",
    "RtpModule",
    "rtp_module",
    "Qwen2VLForConditionalGeneration",
    "Qwen2ForCausalLM",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3VLForConditionalGeneration",
    "DeepSeekV32ForCausalLM",
    "glm_5",
    "Qwen3VLMoeForConditionalGeneration",
    "Glm4MoeForCausalLM",
    "ChatGLMForCausalLM",
    "Qwen2MTPForCausalLM",
    "DeepSeekV32MTPForCausalLM",
]
