import logging
import time
from typing import Dict, Optional

from typing_extensions import override

from rtp_llm.async_decoder_engine.base_engine import BaseEngine
from rtp_llm.async_decoder_engine.base_grammar_backend import (
    InvalidGrammarObject,
    create_grammar_backend,
)
from rtp_llm.config.engine_config import EngineConfig
from rtp_llm.frontend.token_processor import TokenProcessor
from rtp_llm.models.base_model import BaseModel
from rtp_llm.models.propose_model.propose_model import ProposeModel
from rtp_llm.ops import TaskType
from rtp_llm.ops.rtp_llm.rtp_llm_op import RtpLLMOp
from rtp_llm.utils.mm_process_engine import MMProcessEngine
from rtp_llm.utils.time_util import timer_wrapper


class LanguageCppEngine(BaseEngine):
    def __init__(
        self,
        model: BaseModel,
        engine_config: EngineConfig,
        world_info=None,
        propose_model: Optional[ProposeModel] = None,
    ) -> None:
        """Initialize RPCEngine with model and engine configuration.

        Args:
            model: BaseModel instance
            engine_config: EngineConfig instance containing engine and parallelism configs
            world_info: Optional WorldInfo instance from DistributedServer (used for HTTP server)
            propose_model: Optional propose model for speculative decoding
        """
        self.model = model
        self.propose_model = propose_model
        self.tokenizer = model.tokenizer
        self.world_info = world_info
        # BaseModel no longer has config attribute, use model_config instead
        self.config = model.model_config
        self.token_processor = TokenProcessor(
            self.tokenizer, self.model.model_config.special_tokens
        )
        if self.model.is_multimodal():
            self.mm_engine = MMProcessEngine(self.model, self.model.vit_config)
        else:
            self.mm_engine = None

        tokenizer = self.model.tokenizer.tokenizer
        vocab_size = self.config.vocab_size
        grammar_config = engine_config.grammar_config
        from rtp_llm.async_decoder_engine.grammar_cache import resolve_grammar_cache_dir

        self.grammar_backend = create_grammar_backend(
            grammar_backend=grammar_config.grammar_backend,
            constrained_json_disable_any_whitespace=grammar_config.constrained_json_disable_any_whitespace,
            reasoning_parser=grammar_config.reasoning_parser,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=None,
            cache_dir=resolve_grammar_cache_dir(grammar_config.cache_dir),
            debug_log=grammar_config.debug_log,
        )
        if self.grammar_backend is not None:
            # Bind class handle once in Python creation path.
            self.grammar_backend._invalid_grammar_cls = InvalidGrammarObject
        self.rtp_llm_op_ = RtpLLMOp(
            engine_config,
            model,
            self.mm_engine,
            propose_model,
            self.token_processor,
            self.grammar_backend,
        )

    @timer_wrapper(description="start async engine")
    @override
    def _start(self) -> None:
        start_time = time.time()
        self.rtp_llm_op_.start()
        consume_s = time.time() - start_time
        logging.info(f"start rtp_llm_op_ took {consume_s:.2f}s")

        # Start HTTP server for language model tasks
        if (
            self.config.task_type == TaskType.LANGUAGE_MODEL
            and self.world_info is not None
        ):
            self.rtp_llm_op_.ft_op.start_http_server(
                self.model.model_weights_loader,
                self.world_info,
                self.tokenizer,
                None,  # chat_renderer is not needed for HTTP server startup
            )

    @override
    def _stop(self) -> None:
        self.rtp_llm_op_.stop()
