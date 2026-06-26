import logging

from rtp_llm.models_py.model_loader import LoadConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule, rtp_module
from rtp_llm.models_py.registry import MODEL_REGISTRY, get_model_class, register_model

_logger = logging.getLogger(__name__)

# Each model import+registration is wrapped in try/except so that one failure
# doesn't prevent ALL other models from registering.
_IMPORT_ERRORS: dict = {}


def _safe_register(model_type: str, module_path: str, class_name: str):
    """Import and register a model class, logging errors instead of crashing."""
    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        register_model(model_type)(cls)
        return cls
    except Exception as e:
        _IMPORT_ERRORS[model_type] = str(e)
        _logger.warning(
            "Failed to register model '%s' (from %s.%s): %s",
            model_type,
            module_path,
            class_name,
            e,
        )
        return None


# --- Model registrations ---
DeepSeekV32ForCausalLM = _safe_register(
    "deepseek_v32",
    "rtp_llm.models_py.new_models.deepseek_v3",
    "DeepSeekV32ForCausalLM",
)

Qwen2VLForConditionalGeneration = _safe_register(
    "qwen2_vl",
    "rtp_llm.models_py.new_models.qwen2_vl",
    "Qwen2VLForConditionalGeneration",
)

# Qwen2ForCausalLM lives inside qwen2_vl.language
Qwen2ForCausalLM = _safe_register(
    "qwen_2",
    "rtp_llm.models_py.new_models.qwen2_vl.language",
    "Qwen2ForCausalLM",
)

Qwen3ForCausalLM = _safe_register(
    "qwen_3",
    "rtp_llm.models_py.new_models.qwen3",
    "Qwen3ForCausalLM",
)

# qwen_3_tool uses same class as qwen_3
if Qwen3ForCausalLM is not None:
    try:
        register_model("qwen_3_tool")(Qwen3ForCausalLM)
    except Exception:
        pass

Qwen3MoeForCausalLM = _safe_register(
    "qwen_3_moe",
    "rtp_llm.models_py.new_models.qwen3_moe",
    "Qwen3MoeForCausalLM",
)

# qwen3_coder_moe uses same class as qwen_3_moe
if Qwen3MoeForCausalLM is not None:
    try:
        register_model("qwen3_coder_moe")(Qwen3MoeForCausalLM)
    except Exception:
        pass

Qwen3VLForConditionalGeneration = _safe_register(
    "qwen3_vl",
    "rtp_llm.models_py.new_models.qwen3_vl",
    "Qwen3VLForConditionalGeneration",
)

Qwen3VLMoeForConditionalGeneration = _safe_register(
    "qwen3_vl_moe",
    "rtp_llm.models_py.new_models.qwen3_vl_moe",
    "Qwen3VLMoeForConditionalGeneration",
)

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
    "Qwen3VLMoeForConditionalGeneration",
]
