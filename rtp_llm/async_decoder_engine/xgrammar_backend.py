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
"""Constrained decoding with xgrammar backend."""

import dataclasses
import json
import logging
import os
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
from xgrammar import (
    CompiledGrammar,
    GrammarCompiler,
    GrammarMatcher,
    StructuralTag,
    StructuralTagItem,
    TokenizerInfo,
    allocate_token_bitmask,
)

from rtp_llm.async_decoder_engine.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    GrammarStats,
    InvalidGrammarObject,
)
from rtp_llm.async_decoder_engine.grammar_cache import GrammarFileCache
from rtp_llm.models_py.triton_kernels.grammar.bitmask_ops import (
    apply_token_bitmask_inplace_triton,
)

logger = logging.getLogger(__name__)
MAX_ROLLBACK_TOKENS = 200


def is_legacy_structural_tag(structural_tag: Dict) -> bool:
    return "structures" in structural_tag and "triggers" in structural_tag


class XGrammarGrammar(BaseGrammarObject):

    _matcher_pool: Dict[int, List[GrammarMatcher]] = {}
    _POOL_CAP = 32

    def __init__(
        self,
        matcher: GrammarMatcher,
        vocab_size: int,
        ctx: CompiledGrammar,
        override_stop_tokens: Optional[Union[List[int], int]],
        key_string: Optional[str] = None,  # TODO (sk): for debugging, remove later
        grammar_stats: Optional[GrammarStats] = GrammarStats(),
        debug_log: bool = False,
    ) -> None:
        super().__init__()
        self.matcher = matcher
        self.vocab_size = vocab_size
        self.ctx = ctx
        self.override_stop_tokens = override_stop_tokens
        self.accepted_tokens = []
        self.key_string = key_string
        self.grammar_stats = grammar_stats
        self.debug_log = debug_log

    def accept_token(self, token: int):
        if not self.is_terminated():
            self.current_token = token
            terminated_before = self.matcher.is_terminated()
            accepted = self.matcher.accept_token(token)
            if not accepted:
                raise ValueError(
                    f"Tokens not accepted: {token}\n"
                    f"Accepted tokens: {self.accepted_tokens}\n"
                    f"Key string: {self.key_string}"
                )
            else:
                self.accepted_tokens.append(token)
                terminated_after = self.matcher.is_terminated()
                log = logger.info if self.debug_log and len(self.accepted_tokens) <= 12 else logger.debug
                log(
                    "[xgrammar accept_token] key=%s token=%d n_accepted=%d "
                    "terminated=%s->%s finished=%s",
                    (self.key_string or "?")[:60],
                    token,
                    len(self.accepted_tokens),
                    terminated_before,
                    terminated_after,
                    self._finished,
                )
        else:
            logger.debug(
                "[xgrammar accept_token] SKIP (already terminated), token=%d, "
                "n_accepted=%d",
                token,
                len(self.accepted_tokens),
            )

    def rollback(self, k: int):
        self.matcher.rollback(k)
        self.accepted_tokens = self.accepted_tokens[:-k]

    def is_terminated(self):
        return self.matcher.is_terminated()

    def allocate_vocab_mask(
        self, vocab_size: int, batch_size: int, device
    ) -> torch.Tensor:
        return allocate_token_bitmask(batch_size, vocab_size)

    def fill_vocab_mask(self, vocab_mask: torch.Tensor, idx: int) -> None:
        logger.debug(
            "[xgrammar fill_vocab_mask] row=%d, terminated=%s, finished=%s, "
            "n_accepted=%d, bitmask_shape=%s",
            idx,
            self.matcher.is_terminated(),
            self._finished,
            len(self.accepted_tokens),
            list(vocab_mask.shape),
        )
        self.matcher.fill_next_token_bitmask(vocab_mask, idx)

    @staticmethod
    def move_vocab_mask(vocab_mask: torch.Tensor, device) -> torch.Tensor:
        return vocab_mask.to(device, non_blocking=True)

    def apply_vocab_mask(self, logits: torch.Tensor, vocab_mask: torch.Tensor) -> None:
        batch_size = logits.shape[0]
        logits_width = logits.shape[1]
        bitmask_width = (
            vocab_mask.shape[1] if vocab_mask.ndim > 1 else vocab_mask.shape[0]
        )

        logger.debug(
            "[xgrammar apply_vocab_mask] batch=%d, logits_width=%d, "
            "bitmask_width=%d (covers %d tokens), device=%s",
            batch_size,
            logits_width,
            bitmask_width,
            bitmask_width * 32,
            logits.device,
        )

        if logits.device.type == "cuda":
            apply_token_bitmask_inplace_triton(logits, vocab_mask)
        else:
            raise RuntimeError(f"Unsupported device: {logits.device.type}")

    def recycle(self) -> None:
        if self.matcher is None:
            return
        pool = XGrammarGrammar._matcher_pool
        ctx_id = id(self.ctx)
        bucket = pool.get(ctx_id)
        if bucket is None:
            bucket = []
            pool[ctx_id] = bucket
        if len(bucket) < XGrammarGrammar._POOL_CAP:
            self.matcher.reset()
            bucket.append(self.matcher)
        self.matcher = None

    def copy(self):
        pool = XGrammarGrammar._matcher_pool
        ctx_id = id(self.ctx)
        bucket = pool.get(ctx_id)
        if bucket:
            matcher = bucket.pop()
        else:
            matcher = GrammarMatcher(
                self.ctx,
                max_rollback_tokens=MAX_ROLLBACK_TOKENS,
                override_stop_tokens=self.override_stop_tokens,
            )
        if grammar_stats := self.grammar_stats:
            grammar_stats = dataclasses.replace(
                grammar_stats, is_cache_hit=True, tree_traversal_time=[]
            )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            self.ctx,
            self.override_stop_tokens,
            self.key_string,
            grammar_stats,
            debug_log=self.debug_log,
        )

    def try_jump_forward(self, tokenizer) -> Optional[Tuple[List[int], str]]:
        s = self.matcher.find_jump_forward_string()
        if s:
            return [], s
        return None

    def jump_forward_str_state(self, helper: Tuple[List[int], str]) -> Tuple[str, int]:
        _, data = helper
        return data, -1

    def jump_and_retokenize(
        self, old_output_ids: List[int], new_output_ids: List[int], next_state: int
    ):
        k = 0
        for i, old_id in enumerate(old_output_ids):
            if old_id == new_output_ids[i]:
                k = i + 1
            else:
                break

        # rollback to the last token that is the same
        if k < len(old_output_ids):
            self.matcher.rollback(len(old_output_ids) - k)

        for i in range(k, len(new_output_ids)):
            assert self.matcher.accept_token(new_output_ids[i])

    def __repr__(self):
        return f"XGrammarGrammar({self.key_string=}, {self.accepted_tokens=}, {self.current_token=})"


class TokenizerNotSupportedError(Exception):
    """Raised when tokenizer is not supported by XGrammar backend."""

    pass


class XGrammarGrammarBackend(BaseGrammarBackend):
    def __init__(
        self,
        tokenizer,
        vocab_size: int,
        model_eos_token_ids: Optional[List[int]] = None,
        any_whitespace: bool = True,
        cache_dir: Optional[str] = None,
        debug_log: bool = False,
    ):
        super().__init__()
        self.debug_log = debug_log

        if hasattr(tokenizer, "init_xgrammar"):
            tokenizer_info, override_stop_tokens = tokenizer.init_xgrammar()

            if tokenizer_info is None:
                raise TokenizerNotSupportedError(
                    f"Tokenizer type {type(tokenizer).__name__} is not supported by XGrammar"
                )
        else:
            try:
                tokenizer_info = TokenizerInfo.from_huggingface(
                    tokenizer, vocab_size=vocab_size, stop_token_ids=model_eos_token_ids
                )
                override_stop_tokens = None
            except Exception as e:
                raise TokenizerNotSupportedError(
                    f"Failed to create XGrammar TokenizerInfo from tokenizer: {e}"
                )

        self.grammar_compiler = GrammarCompiler(tokenizer_info=tokenizer_info)
        self.vocab_size = vocab_size
        self.override_stop_tokens = override_stop_tokens
        self.any_whitespace = any_whitespace

        self.file_cache: Optional[GrammarFileCache] = None
        if cache_dir:
            try:
                self.file_cache = GrammarFileCache(cache_dir, tokenizer_info)
            except Exception as e:
                logger.debug("[xgrammar backend_init] file cache init failed: %s", e)

        logger.debug(
            "[xgrammar backend_init] pid=%d, vocab_size=%d, any_whitespace=%s, "
            "override_stop_tokens=%s, model_eos_token_ids=%s, tokenizer_type=%s, "
            "file_cache=%s",
            os.getpid(),
            vocab_size,
            any_whitespace,
            override_stop_tokens,
            model_eos_token_ids,
            type(tokenizer).__name__,
            "enabled" if self.file_cache else "disabled",
        )

    @staticmethod
    def _sanitize_structural_format(structural_format):
        """Recursively replace missing json_schema fields with an empty schema."""
        if not isinstance(structural_format, dict):
            return

        fmt_type = structural_format.get("type")
        if fmt_type in {"json_schema", "qwen_xml_parameter"}:
            if structural_format.get("json_schema") is None:
                structural_format["json_schema"] = {}

        if fmt_type == "tag":
            XGrammarGrammarBackend._sanitize_structural_format(
                structural_format.get("content")
            )
        elif fmt_type in {"sequence", "or"}:
            for element in structural_format.get("elements", []):
                XGrammarGrammarBackend._sanitize_structural_format(element)
        elif fmt_type in {"triggered_tags", "tags_with_separator"}:
            for tag in structural_format.get("tags", []):
                XGrammarGrammarBackend._sanitize_structural_format(tag)

    @staticmethod
    def _sanitize_structural_tag_structures(structural_tag: Dict) -> None:
        for structure in structural_tag.get("structures", []):
            if structure.get("schema") is None:
                structure["schema"] = {}

    def _try_file_cache(
        self, key_type: str, key_string: str
    ) -> Optional[CompiledGrammar]:
        if self.file_cache is None:
            return None
        return self.file_cache.get(key_type, key_string)

    def _store_file_cache(
        self, key_type: str, key_string: str, ctx: CompiledGrammar
    ) -> None:
        if self.file_cache is not None:
            self.file_cache.put(key_type, key_string, ctx)

    def _from_context(
        self, ctx: CompiledGrammar, key_string: str, grammar_stats: GrammarStats
    ) -> XGrammarGrammar:
        matcher = GrammarMatcher(
            ctx,
            max_rollback_tokens=MAX_ROLLBACK_TOKENS,
            override_stop_tokens=self.override_stop_tokens,
        )
        return XGrammarGrammar(
            matcher,
            self.vocab_size,
            ctx,
            self.override_stop_tokens,
            key_string,
            grammar_stats,
            debug_log=self.debug_log,
        )

    def dispatch_json(self, key_string: str) -> BaseGrammarObject:
        pid = os.getpid()
        logger.debug(
            "[xgrammar dispatch_json] pid=%d, schema_len=%d, schema=%.200s",
            pid,
            len(key_string),
            key_string,
        )
        ctx = self._try_file_cache("json", key_string)
        if ctx is not None:
            logger.info(
                "[xgrammar dispatch_json] pid=%d, FILE_CACHE_HIT, skipping compile",
                pid,
            )
            return self._from_context(
                ctx, key_string, GrammarStats(dispatch_type="json", is_cache_hit=True)
            )
        logger.info(
            "[xgrammar dispatch_json] pid=%d, FILE_CACHE_MISS, compiling...", pid
        )
        try:
            t0 = time.perf_counter()
            if key_string == "$$ANY$$":
                ctx = self.grammar_compiler.compile_builtin_json_grammar()
            else:
                ctx = self.grammar_compiler.compile_json_schema(
                    schema=key_string, any_whitespace=self.any_whitespace
                )
            compile_ms = (time.perf_counter() - t0) * 1000

        except (RuntimeError, json.decoder.JSONDecodeError, UnicodeDecodeError) as e:
            logger.error(
                "[xgrammar dispatch_json] pid=%d, compile FAILED: %s, schema=%.200s",
                pid,
                e,
                key_string,
            )
            return InvalidGrammarObject(str(e))
        logger.info(
            "[xgrammar dispatch_json] pid=%d, COMPILED, compile_ms=%.2f, "
            "vocab_size=%d, memory_size_bytes=%d",
            pid,
            compile_ms,
            self.vocab_size,
            ctx.memory_size_bytes,
        )
        self._store_file_cache("json", key_string, ctx)
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="json"))

    def dispatch_ebnf(self, key_string: str) -> BaseGrammarObject:
        pid = os.getpid()
        logger.debug(
            "[xgrammar dispatch_ebnf] pid=%d, ebnf_len=%d", pid, len(key_string)
        )
        ctx = self._try_file_cache("ebnf", key_string)
        if ctx is not None:
            logger.info("[xgrammar dispatch_ebnf] pid=%d, FILE_CACHE_HIT", pid)
            return self._from_context(
                ctx, key_string, GrammarStats(dispatch_type="ebnf", is_cache_hit=True)
            )
        logger.info(
            "[xgrammar dispatch_ebnf] pid=%d, FILE_CACHE_MISS, compiling...", pid
        )
        try:
            t0 = time.perf_counter()
            ctx = self.grammar_compiler.compile_grammar(key_string)
            compile_ms = (time.perf_counter() - t0) * 1000
        except RuntimeError as e:
            logger.error("[xgrammar dispatch_ebnf] pid=%d, compile FAILED: %s", pid, e)
            return InvalidGrammarObject(str(e))
        logger.info(
            "[xgrammar dispatch_ebnf] pid=%d, COMPILED, compile_ms=%.2f",
            pid,
            compile_ms,
        )
        self._store_file_cache("ebnf", key_string, ctx)
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="ebnf"))

    def dispatch_regex(self, key_string: str) -> BaseGrammarObject:
        pid = os.getpid()
        logger.debug(
            "[xgrammar dispatch_regex] pid=%d, pattern_len=%d, pattern=%.100s",
            pid,
            len(key_string),
            key_string,
        )
        ctx = self._try_file_cache("regex", key_string)
        if ctx is not None:
            logger.info("[xgrammar dispatch_regex] pid=%d, FILE_CACHE_HIT", pid)
            return self._from_context(
                ctx, key_string, GrammarStats(dispatch_type="regex", is_cache_hit=True)
            )
        logger.info(
            "[xgrammar dispatch_regex] pid=%d, FILE_CACHE_MISS, compiling...", pid
        )
        try:
            t0 = time.perf_counter()
            ctx = self.grammar_compiler.compile_regex(key_string)
            compile_ms = (time.perf_counter() - t0) * 1000
        except RuntimeError as e:
            logger.error("[xgrammar dispatch_regex] pid=%d, compile FAILED: %s", pid, e)
            return InvalidGrammarObject(str(e))
        logger.info(
            "[xgrammar dispatch_regex] pid=%d, COMPILED, compile_ms=%.2f",
            pid,
            compile_ms,
        )
        self._store_file_cache("regex", key_string, ctx)
        return self._from_context(ctx, key_string, GrammarStats(dispatch_type="regex"))

    def dispatch_structural_tag(self, key_string: str) -> BaseGrammarObject:
        pid = os.getpid()
        logger.debug(
            "[xgrammar dispatch_structural_tag] pid=%d, tag_len=%d",
            pid,
            len(key_string),
        )
        ctx = self._try_file_cache("structural_tag", key_string)
        if ctx is not None:
            logger.info(
                "[xgrammar dispatch_structural_tag] pid=%d, FILE_CACHE_HIT", pid
            )
            return self._from_context(
                ctx,
                key_string,
                GrammarStats(dispatch_type="structural_tag", is_cache_hit=True),
            )
        logger.info(
            "[xgrammar dispatch_structural_tag] pid=%d, FILE_CACHE_MISS, compiling...",
            pid,
        )
        try:
            structural_tag = json.loads(key_string)
            is_legacy = is_legacy_structural_tag(structural_tag)
            logger.debug("[xgrammar dispatch_structural_tag] is_legacy=%s", is_legacy)
            if is_legacy:
                self._sanitize_structural_tag_structures(structural_tag)
                tags = [
                    StructuralTagItem(
                        begin=structure["begin"],
                        schema=json.dumps(structure["schema"]),
                        end=structure["end"],
                    )
                    for structure in structural_tag["structures"]
                ]
                new_tag = StructuralTag.from_legacy_structural_tag(
                    tags, structural_tag["triggers"]
                )
                new_tag.format.at_least_one = structural_tag.get("at_least_one", False)
                ctx = self.grammar_compiler.compile_structural_tag(new_tag)
            else:
                format_dict = structural_tag.get("format")
                if isinstance(format_dict, dict):
                    self._sanitize_structural_format(format_dict)
                    structural_tag["format"] = format_dict
                    key_string = json.dumps(structural_tag)
                ctx = self.grammar_compiler.compile_structural_tag(key_string)
        except (RuntimeError, json.decoder.JSONDecodeError) as e:
            logger.error("[xgrammar dispatch_structural_tag] compile failed: %s", e)
            return InvalidGrammarObject(str(e))
        logger.info("[xgrammar dispatch_structural_tag] pid=%d, COMPILED", pid)
        self._store_file_cache("structural_tag", key_string, ctx)
        return self._from_context(
            ctx, key_string, GrammarStats(dispatch_type="structural_tag")
        )

    def reset(self):
        super().reset()
        self.grammar_compiler.clear_cache()
