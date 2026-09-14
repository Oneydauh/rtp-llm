import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Optional

import torch

from rtp_llm.config.model_config import ModelConfig
from rtp_llm.device.device_type import DeviceType, get_device_type
from rtp_llm.models_py.model_desc.block_map import (
    get_attention_inputs_value,
    select_attention_inputs_for_tag,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.modules import AttnImplFactory
from rtp_llm.models_py.modules.factory.attention.attn_factory import AttentionImpl
from rtp_llm.ops import DeviceResourceConfig
from rtp_llm.ops.compute_ops import (
    KVCache,
    PyModelInitResources,
    PyModelInputs,
    PyModelOutputs,
)

if TYPE_CHECKING:
    from rtp_llm.model_loader.model_weight_info import ModelWeights


def required_config_value(config: Any, *names: str) -> Any:
    for name in names:
        value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if value is not None:
            return value
    raise ValueError(f"Model config requires one of {names}")


class GptModelBase(RtpModule):
    def __init__(
        self,
        config: ModelConfig,
        parallelism_config,
        weight: Optional["ModelWeights"] = None,
        max_generate_batch_size: int = 0,
        fmha_config=None,  # Optional FMHAConfig
        py_hw_kernel_config=None,  # Optional HWKernelConfig
        device_resource_config: Optional[
            DeviceResourceConfig
        ] = None,  # Optional DeviceResourceConfig
    ) -> None:
        super().__init__()
        self.config = config
        self.parallelism_config = parallelism_config
        self.weight = weight
        self.fmha_config = fmha_config
        self.py_hw_kernel_config = py_hw_kernel_config
        self.micro_batch_size: int = (
            1
            if device_resource_config
            and device_resource_config.enable_layer_micro_batch == 0
            else 2
        )
        self.layer_num = required_config_value(
            config, "num_layers", "num_hidden_layers"
        )
        self.vocab_size = required_config_value(config, "vocab_size")

        self.kv_cache: Optional[KVCache] = None
        self.device_type: DeviceType = get_device_type()

    def initialize(self, init_resource: PyModelInitResources) -> bool:
        self.kv_cache = init_resource.kv_cache
        log_stats = getattr(init_resource, "eplb_log_stats", None)
        gpu_loads = getattr(init_resource, "eplb_gpu_loads", None)
        if (
            isinstance(log_stats, torch.Tensor)
            and isinstance(gpu_loads, torch.Tensor)
            and log_stats.numel() > 0
            and gpu_loads.numel() > 0
        ):
            for module in self.modules():
                bind = getattr(module, "bind_eplb_stats", None)
                layer_idx = getattr(module, "layer_idx", None)
                if callable(bind) and isinstance(layer_idx, int):
                    bind(log_stats[layer_idx], gpu_loads[layer_idx])
        if self.kv_cache is not None:
            num_layers = self.kv_cache.layer_count
            layer0_caches = (
                self.kv_cache.get_layer_cache_groups(0) if num_layers > 0 else []
            )
            layer0_shapes = [cache.kv_cache_base.shape for cache in layer0_caches]
            layer0_scale_count = sum(
                cache.kv_scale_base is not None and cache.kv_scale_base.numel() > 0
                for cache in layer0_caches
            )
            logging.info(
                f"GptModelBase initialized with "
                f"num_kv_layers={num_layers}, "
                f"layer0_kv_cache_shapes={layer0_shapes}, "
                f"layer0_scale_groups={layer0_scale_count}, "
            )
        return True

    def apply_eplb_plan(
        self,
        layer_id: int,
        phy2log,
        logic_expert_cnt,
        log2phy,
        moe_weight_1,
        moe_weight_2,
        moe_scale_1,
        moe_scale_2,
    ) -> bool:
        from rtp_llm.utils.model_weight import W

        matches = [
            module
            for module in self.modules()
            if getattr(module, "layer_idx", None) == layer_id
            and callable(getattr(module, "apply_eplb_plan", None))
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected one EPLB-capable MoE module for layer {layer_id}, "
                f"found {len(matches)}"
            )
        matches[0].apply_eplb_plan(
            phy2log,
            logic_expert_cnt,
            log2phy,
            {
                W.moe_w1: moe_weight_1,
                W.moe_w2: moe_weight_2,
                W.moe_s1: moe_scale_1,
                W.moe_s2: moe_scale_2,
            },
            defer_commit=True,
        )
        return True

    def supports_eplb_weight_install(self) -> bool:
        return any(
            bool(getattr(module, "_supports_newloader_eplb", False))
            and bool(getattr(module, "_dynamic_eplb_enabled", False))
            for module in self.modules()
        )

    def prepare_fmha_impl(
        self, inputs: PyModelInputs, is_cuda_graph: bool = False
    ) -> AttentionImpl | dict[str, AttentionImpl]:
        attention_inputs = get_attention_inputs_value(inputs)
        if isinstance(attention_inputs, Mapping):
            fmha_group_tags = self._get_fmha_group_tags()
            selected_group_inputs = (
                attention_inputs.items()
                if fmha_group_tags is None
                else (
                    (tag, select_attention_inputs_for_tag(attention_inputs, tag))
                    for tag in fmha_group_tags
                )
            )
            return {
                tag: AttnImplFactory.get_fmha_impl(
                    self.config,
                    self.parallelism_config,
                    self.weight,
                    group_inputs,
                    self.fmha_config,
                    is_cuda_graph,
                )
                for tag, group_inputs in selected_group_inputs
            }
        return AttnImplFactory.get_fmha_impl(
            self.config,
            self.parallelism_config,
            self.weight,
            attention_inputs,
            self.fmha_config,
            is_cuda_graph,
        )

    def _get_fmha_group_tags(self) -> Optional[list[str]]:
        """Model hook: None means every attention-input tag requires FMHA."""
        return None

    def forward(self, inputs: PyModelInputs, fmha_impl: Any = None) -> PyModelOutputs:
        raise NotImplementedError("forward method must be implemented in subclass")
