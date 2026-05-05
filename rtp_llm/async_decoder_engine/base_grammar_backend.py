# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""The baseclass of a backend for grammar-guided constrained decoding.

Threading model (post-refactor):
  The C++ `GrammarManager` owns the compile worker pool and all queuing logic.
  Python here is only a passive library: the manager calls `get_cached` and
  `compile_now` synchronously from its worker threads (each holding GIL).
  No Python `Future` / `ThreadPoolExecutor` is involved anymore.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


@dataclass
class GrammarStats:
    compilation_time: Optional[float] = None
    schema_count: Optional[int] = None
    ebnf_size: Optional[int] = None
    is_cache_hit: bool = False
    is_grammar_aborted: bool = False
    tree_traversal_time: List[float] = field(default_factory=list)
    dispatch_type: Optional[str] = None
    num_timeout: int = 0


class BaseGrammarObject:

    def __init__(self):
        self._finished = False
        self.grammar_stats = None
        self.current_token = None

    def maybe_init_reasoning(self, reasoning: bool):
        pass

    def accept_token(self, token: int) -> None:
        """
        Accept a token in the grammar.
        """
        raise NotImplementedError()

    def rollback(self, k: int):
        raise NotImplementedError()

    def is_terminated(self):
        return False

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        raise NotImplementedError()

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        raise NotImplementedError()

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        raise NotImplementedError()

    @staticmethod
    def apply_vocab_mask(logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        raise NotImplementedError()

    def copy(self) -> "BaseGrammarObject":
        return self

    def recycle(self) -> None:
        pass

    @property
    def finished(self):
        return self._finished

    @finished.setter
    def finished(self, finished):
        self._finished = finished

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        """
        Try to jump forward in the grammar.

        Returns:
            A jump forward helper which may be used in `jump_forward_str_state`.
            None if the jump forward is not possible.
        """
        raise NotImplementedError()

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        """
        Jump forward for the grammar.

        Returns:
            A tuple of the jump forward string and the next state of the grammar
            (which can be used in `jump_and_retokenize` if needed).
        """
        raise NotImplementedError()

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ) -> None:
        """
        Jump forward occurs, and update the grammar state if needed.
        """
        raise NotImplementedError()


class InvalidGrammarObject(BaseGrammarObject):
    """Represents a grammar that failed to compile, carrying the original error message."""

    def __init__(self, error_message: str = "Unknown grammar error"):
        super().__init__()
        self.error_message = error_message

    def __repr__(self):
        return f"InvalidGrammarObject(error_message={self.error_message!r})"


class BaseGrammarBackend:
    # Exposed so the C++ GrammarManager can construct an InvalidGrammarObject
    # (e.g. to cache a failure marker on compile timeout) without importing
    # this module directly.
    _invalid_grammar_cls = InvalidGrammarObject

    def __init__(self):
        self.cache: Dict[Tuple[str, str], BaseGrammarObject] = {}

    def _not_supported(self, key_type: str, key_string: str) -> BaseGrammarObject:
        logger.warning(f"Skip unsupported {key_type=}, {key_string=}")
        return InvalidGrammarObject()

    def dispatch_fallback(self, key_type: str, key_string: str) -> BaseGrammarObject:
        """
        This function should not be reached in any case.
        """
        raise ValueError(f"Invalid key_type: {key_type}={key_string}")

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("json", key_string)

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("regex", key_string)

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("ebnf", key_string)

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        return self._not_supported("structural_tag", key_string)

    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        import os

        pid = os.getpid()
        s = time.perf_counter()
        key_type, key_string = key
        logger.debug(
            "[grammar_backend] _init_value_dispatch pid=%d, key_type=%s, "
            "key_len=%d, require_reasoning=%s",
            pid,
            key_type,
            len(key_string),
            require_reasoning,
        )
        if key_type == "json":
            grammar = self.dispatch_json(key_string)
        elif key_type == "regex":
            grammar = self.dispatch_regex(key_string)
        elif key_type == "ebnf":
            grammar = self.dispatch_ebnf(key_string)
        elif key_type == "structural_tag":
            grammar = self.dispatch_structural_tag(key_string)
        else:
            grammar = self.dispatch_fallback(key_type, key_string)

        elapsed_ms = (time.perf_counter() - s) * 1000
        if grammar is not None and grammar.grammar_stats is not None:
            grammar.grammar_stats.compilation_time = time.perf_counter() - s
        logger.debug(
            "[grammar_backend] _init_value_dispatch DONE pid=%d, key_type=%s, "
            "elapsed_ms=%.2f, grammar_type=%s",
            pid,
            key_type,
            elapsed_ms,
            type(grammar).__name__,
        )
        return grammar

    def get_cached(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> Optional[BaseGrammarObject]:
        """Return a copy of the cached grammar, or None.

        Called synchronously from the C++ GrammarManager (GIL held) during
        `process_req_with_grammar` fast path. A copy is returned so the caller
        can mutate the grammar without touching the cache entry.
        """
        import os

        pid = os.getpid()
        value = self.cache.get(key)
        if value is None:
            logger.info(
                "[grammar_backend] get_cached pid=%d, key_type=%s, MEMORY_CACHE_MISS",
                pid,
                key[0],
            )
            return None
        copied_value = value.copy()
        copied_value.maybe_init_reasoning(require_reasoning)
        logger.info(
            "[grammar_backend] get_cached pid=%d, key_type=%s, MEMORY_CACHE_HIT",
            pid,
            key[0],
        )
        return copied_value

    def compile_now(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        """Synchronous compile entry, called from C++ worker thread (GIL held).

        Returns a freshly compiled BaseGrammarObject (or InvalidGrammarObject
        on failure). The caller is responsible for inserting into cache via
        `set_cache` if desired.
        """
        return self._init_value_dispatch(key, require_reasoning)

    def set_cache(self, key: Tuple[str, str], value: BaseGrammarObject):
        self.cache[key] = value

    def reset(self):
        self.cache.clear()


GRAMMAR_BACKEND_REGISTRY = {}


def register_grammar_backend(name, init_func):
    GRAMMAR_BACKEND_REGISTRY[name] = init_func


def create_grammar_backend(
    grammar_backend: str,
    constrained_json_disable_any_whitespace: bool,
    reasoning_parser: bool,
    tokenizer,
    vocab_size: int,
    eos_token_ids: Optional[set] = None,
    cache_dir: Optional[str] = None,
    debug_log: bool = False,
) -> Optional[BaseGrammarBackend]:
    name = grammar_backend

    if name not in (None, "", "none") and not torch.cuda.is_available():
        logger.warning(
            "Grammar backend '%s' requested but CUDA is not available; "
            "disabling grammar. Structured outputs (JSON schema, regex, EBNF) "
            "will not be enforced.",
            name,
        )
        return None

    # Custom grammar backend has the highest priority
    if name in GRAMMAR_BACKEND_REGISTRY:
        return GRAMMAR_BACKEND_REGISTRY[name](
            grammar_backend=grammar_backend,
            constrained_json_disable_any_whitespace=constrained_json_disable_any_whitespace,
            reasoning_parser=reasoning_parser,
            tokenizer=tokenizer,
            vocab_size=vocab_size,
            eos_token_ids=eos_token_ids,
        )
    # Default grammar backends
    if name == "xgrammar":
        from rtp_llm.async_decoder_engine.xgrammar_backend import (
            TokenizerNotSupportedError,
            XGrammarGrammarBackend,
        )

        # Convert Set[int] to List[int] if needed
        eos_list = list(eos_token_ids) if eos_token_ids else None

        try:
            grammar_backend = XGrammarGrammarBackend(
                tokenizer,
                vocab_size=vocab_size,
                model_eos_token_ids=eos_list,
                any_whitespace=not constrained_json_disable_any_whitespace,
                cache_dir=cache_dir,
                debug_log=debug_log,
            )
        except TokenizerNotSupportedError as e:
            logger.warning(
                f"Grammar backend disabled because tokenizer is not supported by XGrammar: {e}. "
                "Falling back to grammar_backend='none'. "
                "Structured outputs (JSON schema, regex, EBNF) will not be available."
            )
            return None
    elif name == "none":
        return None
    else:
        raise ValueError(f"Invalid grammar backend: {name}")

    logging.info(f"Reasoning parser: {reasoning_parser}")
    if reasoning_parser and hasattr(tokenizer, "think_end_id"):
        from rtp_llm.async_decoder_engine.reasoner_grammar_backend import (
            ReasonerGrammarBackend,
        )

        grammar_backend = ReasonerGrammarBackend(
            grammar_backend, tokenizer.think_end_id
        )

    return grammar_backend
