from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from rtp_llm.config.model_config import ModelConfig
    from rtp_llm.ops import DeviceResourceConfig, ParallelismConfig


class NewLoaderConfigSource(Protocol):
    use_new_loader: Optional[bool]
    require_weight_update: Optional[bool]


_INDEPENDENT_TOPOLOGY_MODEL_TYPES = frozenset(
    {
        "deepseek2",
        "deepseek3",
        "deepseek_v31",
        "deepseek_v32",
        "glm_5",
        "glm4_moe_lite",
        "kimi_k2",
        "qwen_3",
        "qwen_3_moe",
        "qwen_3_tool",
        "qwen3_vl",
        "qwen3_vl_moe",
    }
)


def is_new_loader_enabled(
    model_config: NewLoaderConfigSource, *, default_enabled: bool = False
) -> bool:
    """Resolve an explicit loader override against the model-specific default."""
    if not isinstance(default_enabled, bool):
        raise TypeError("default_enabled must be a bool")
    configured = model_config.use_new_loader
    if configured is None:
        return default_enabled
    if not isinstance(configured, bool):
        raise TypeError("model_config.use_new_loader must be a bool or None")
    return configured


def new_loader_unsupported_reason(
    model_config: "ModelConfig",
    *,
    skip_python_model: bool = False,
    force_cpu_load_weights: bool = False,
    device_resource_config: Optional["DeviceResourceConfig"] = None,
    parallelism_config: Optional["ParallelismConfig"] = None,
) -> Optional[str]:
    """Return why a runtime configuration still requires the legacy loader.

    Shared by language and multimodal loading so automatic routing cannot
    select different loaders for the two halves of one deployment.
    """
    if skip_python_model:
        return "newloader requires the Python model runtime"
    if force_cpu_load_weights:
        return "force_cpu_load_weights is not supported by this newloader slice"
    if model_config.enable_output_vocab_pruning:
        return "output vocabulary pruning is not supported by this newloader slice"
    if model_config.eplb_config.enable_eplb() and model_config.expert_num <= 0:
        return "EPLB requires a NewLoader MoE model with routed experts"
    if model_config.ptuning_path:
        return "p-tuning is not supported by this newloader slice"
    if model_config.lora_infos:
        return "LoRA loading is not supported by this newloader slice"
    if (
        model_config.use_new_loader is None
        and model_config.require_weight_update is None
    ):
        return (
            "the online UpdateWeights policy is undeclared; set "
            "--require_weight_update false to opt in to automatic NewLoader "
            "routing, or true to retain the legacy loader"
        )
    if model_config.require_weight_update:
        if getattr(model_config, "model_type", None) not in ("qwen_2", "qwen_3", "llama"):
            return "online UpdateWeights requires a supported dense Qwen2/Qwen3/Llama NewLoader model"
        if model_config.quant_config is not None:
            return "quantized online UpdateWeights requires a weight/scale update contract; use --use_new_loader false"
    quant_config = model_config.quant_config
    if quant_config is not None:
        runtime_method = quant_config.get_runtime_method_key()
        if not isinstance(runtime_method, str) or not runtime_method.strip():
            return (
                f"quantization config {type(quant_config).__name__} does not "
                "provide a supported NewLoader runtime method"
            )
    if (
        device_resource_config is not None
        and device_resource_config.enable_layer_micro_batch != 0
    ):
        return "layer micro-batch is not supported by this newloader slice"
    if parallelism_config is None:
        return None

    attn_tp = (
        parallelism_config.get_attn_tp_size(),
        parallelism_config.get_attn_tp_rank(),
    )
    ffn_tp = (
        parallelism_config.get_ffn_tp_size(),
        parallelism_config.get_ffn_tp_rank(),
    )
    physical_tp = (parallelism_config.tp_size, parallelism_config.tp_rank)
    supports_independent_topology = (
        getattr(model_config, "model_type", None)
        in _INDEPENDENT_TOPOLOGY_MODEL_TYPES
    )
    if (
        parallelism_config.prefill_cp_config.is_enabled()
        or attn_tp != physical_tp
    ) and not supports_independent_topology:
        return "Context parallelism is not supported by this newloader slice"
    if ffn_tp != attn_tp and not supports_independent_topology:
        return (
            "Independent FFN TP/sequence parallelism is not supported by this "
            "newloader slice"
        )
    if parallelism_config.ffn_disaggregate_config.enable_ffn_disaggregate:
        return "FFN disaggregation is not supported by this newloader slice"
    return None
