"""CUDA FP8 PerBlock quantization strategies"""

from typing import Any

import torch

from rtp_llm.models_py.modules.factory.fused_moe.defs.config_adapter import (
    MoEConfigAdapter,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.priority_attributes import (
    StrategyAttributes,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.quant_config import (
    FusedMoEQuantConfig,
)
from rtp_llm.models_py.modules.factory.fused_moe.defs.strategy_base import MoeStrategy
from rtp_llm.models_py.modules.factory.fused_moe.utils.config_resolver import (
    MoeConfigResolver,
)


class CudaFp8PerBlockNoDPStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single GPU strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_no_dp"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockNoDPMaskedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock No DP Masked strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "fp8_per_block_no_dp_masked")

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor_v2 import (
            DeepGemmMaskedExecutorV2,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlock,
            executor_class=DeepGemmMaskedExecutorV2,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpLowLatencyStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low latency strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_low_latency"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_masked_executor import (
            DeepGemmMaskedExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=DeepGemmMaskedExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockNoDPCutlassFusedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock single-card / pure-TP strategy backed by flashinfer cutlass_fused_moe."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.utils.arch import get_sm

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "fp8_per_block_no_dp_cutlass_fused")
        checker.check(get_sm()[0] == 9)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_fused_moe import (
            CutlassFusedMoeFp8PerBlockExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.pure_tp_router import (
            PureTpRouterFp8PerBlockBf16Passthrough,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=PureTpRouterFp8PerBlockBf16Passthrough,
            executor_class=CutlassFusedMoeFp8PerBlockExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpLowLatencyCutlassFusedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP low-latency strategy backed by flashinfer cutlass_fused_moe."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.utils.arch import get_sm

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "fp8_per_block_ep_low_latency_cutlass_fused")
        checker.check(get_sm()[0] == 9)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_fused_moe import (
            CutlassFusedMoeFp8PerBlockEpLowLatencyExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_low_latency_router import (
            DeepEpLowLatencyRouter,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepEpLowLatencyRouter,
            executor_class=CutlassFusedMoeFp8PerBlockEpLowLatencyExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpNormalCutlassFusedStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal strategy backed by flashinfer cutlass_fused_moe."""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        from rtp_llm.models_py.utils.arch import get_sm

        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(config.moe_strategy == "fp8_per_block_ep_normal_cutlass_fused")
        checker.check(get_sm()[0] == 9)

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.cutlass_fused_moe import (
            CutlassFusedMoeFp8PerBlockEpNormalExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerBlock,
            executor_class=CutlassFusedMoeFp8PerBlockEpNormalExecutor,
            quant_config=quant_config,
        )


class CudaFp8PerBlockEpNormalStrategy(MoeStrategy):
    """CUDA FP8 PerBlock EP normal mode strategy"""

    @classmethod
    def check_conditions(cls, checker: Any, config: MoEConfigAdapter) -> None:
        resolver = MoeConfigResolver()
        quant_method = resolver.get_quant_method(config)
        checker.check(quant_method == "FP8_PER_BLOCK")
        checker.check(
            config.moe_strategy == "fp8_per_block_ep_normal"
            or config.moe_strategy == "auto"
        )

    def get_attributes(self) -> StrategyAttributes:
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.executors.deepgemm_hybrid_executor import (
            DeepGemmHybridExecutor,
        )
        from rtp_llm.models_py.modules.factory.fused_moe.impl.cuda.routers.deepep_normal_router import (
            DeepepNormalRouterFp8PerBlock,
        )

        quant_config = FusedMoEQuantConfig(
            quant_dtype=torch.float8_e4m3fn,
            block_shape=[128, 128],
        )
        return StrategyAttributes(
            router_class=DeepepNormalRouterFp8PerBlock,
            executor_class=DeepGemmHybridExecutor,
            quant_config=quant_config,
        )
