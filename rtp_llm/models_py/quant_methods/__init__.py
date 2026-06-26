# 导入即触发 @register_quant_method / @register_moe_quant_method 注册。
from rtp_llm.models_py.quant_methods.awq import AWQLinearMethod
from rtp_llm.models_py.quant_methods.base import QuantizationConfig, QuantizeMethodBase
from rtp_llm.models_py.quant_methods.fp8 import (
    Fp8BlockDequantLinearMethod,
    Fp8BlockLinearMethod,
    Fp8BlockOnlineLinearMethod,
    Fp8LinearMethod,
    Fp8OnlineLinearMethod,
    Fp8PerChannelOnlineLinearMethod,
)
from rtp_llm.models_py.quant_methods.fp8_moe import Fp8MoEMethod
from rtp_llm.models_py.quant_methods.unquantized import (
    UnquantizedFusedMoEMethod,
    UnquantizedLinearMethod,
)

__all__ = [
    "QuantizeMethodBase",
    "QuantizationConfig",
    "Fp8LinearMethod",
    "Fp8OnlineLinearMethod",
    "Fp8PerChannelOnlineLinearMethod",
    "Fp8BlockOnlineLinearMethod",
    "Fp8BlockLinearMethod",
    "Fp8BlockDequantLinearMethod",
    "Fp8MoEMethod",
    "AWQLinearMethod",
    "UnquantizedLinearMethod",
    "UnquantizedFusedMoEMethod",
]
