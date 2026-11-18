from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from pydantic import BaseModel

from rtp_llm.async_decoder_engine.embedding.interface import EngineInputs, EngineOutputs
from rtp_llm.config.model_config import ModelConfig
from rtp_llm.frontend.tokenizer_factory.tokenizers import BaseTokenizer
from rtp_llm.models_py.downstream_heads import handler_weight_mappings

if TYPE_CHECKING:
    from rtp_llm.ops import EmbeddingCppOutput

"""
用于多种多样的下游任务
"""


class CustomModule(object):
    renderer: "CustomRenderer"
    handler: "CustomHandler"

    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def create_cpp_handler(self) -> Any:
        raise NotImplementedError("not support cpp handler")

    def get_renderer(self, request: Dict[str, Any]) -> "CustomRenderer":
        return self.renderer

    def get_handler(self) -> "CustomHandler":
        return self.handler

    def checkpoint_weight_mappings(self):
        return handler_weight_mappings(self.handler.checkpoint_weight_names())

    def init(self, weight):
        self.handler.init(
            {
                name: weight.get_global_weight(role)
                for role, name in self.checkpoint_weight_mappings()
            }
        )


class CustomHandler(object):
    def __init__(self, config: ModelConfig):
        self.config_ = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.need_post_process = False

    def checkpoint_weight_names(self) -> tuple[str, ...]:
        return ()

    # for cpp
    def init_cpp_handler(self) -> None:
        pass

    def init(self, tensor_map: Dict[str, torch.Tensor]) -> None:
        pass

    # 输入:
    # input_ids: [token_len]
    # hidden_states: [token_len, hidden_size]
    # input_lengths: [batch_size]
    # 输出:
    # [batch_size], 由endpoint格式化返回
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> Union[torch.Tensor, List[Any]]:
        raise NotImplementedError

    # specify required args for extended_forward
    def extend_forward_args(self) -> List[str]:
        return ["input_lengths", "input_ids", "hidden_states"]

    def select_token_position(
        self, input_ids: torch.Tensor, text_tokens_mask: Optional[torch.Tensor] = None
    ) -> int:
        """Select on expanded CPU prompt tokens; -1 opts out. Called once per request."""
        raise NotImplementedError

    # extended_forward
    # input_lengths: [batch_size]
    # input_ids: [token_len]
    # hidden_states: [token_len, hidden_size]
    # moe_gating: list of Optional[Tensor], with length = layer_num, [token_len, expert_num]
    def extend_forward(self, **kwargs: Any) -> Union[torch.Tensor, List[Any]]:
        return self.forward(
            input_lengths=kwargs["input_lengths"],
            input_ids=kwargs["input_ids"],
            hidden_states=kwargs["hidden_states"],
        )

    def post_process(
        self, request: Any, batch_output: EmbeddingCppOutput
    ) -> EmbeddingCppOutput:
        return batch_output


class CustomRenderer(object):
    def __init__(self, config: ModelConfig, tokenizer: BaseTokenizer):
        self.config_ = config
        self.tokenizer_ = tokenizer

    def render_request(self, request_json: Dict[str, Any]) -> BaseModel:
        raise NotImplementedError

    def create_input(self, request: BaseModel) -> EngineInputs:
        raise NotImplementedError

    async def render_response(
        self, request: BaseModel, inputs: EngineInputs, outputs: EngineOutputs
    ) -> Dict[str, Any]:
        raise NotImplementedError

    async def render_log_response(self, response: Dict[str, Any]):
        return response
