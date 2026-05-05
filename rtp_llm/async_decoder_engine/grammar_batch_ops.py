"""Batch grammar operations for constrained decoding hot path.

These functions consolidate per-stream grammar logic into single batch calls,
reducing C++/Python cross-language overhead on the decode hot path.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import torch
from xgrammar import allocate_token_bitmask

from rtp_llm.models_py.triton_kernels.grammar.bitmask_ops import (
    apply_token_bitmask_inplace_triton,
)

logger = logging.getLogger(__name__)

_FILL_PARALLEL_THRESHOLD = 4
_fill_pool = ThreadPoolExecutor(max_workers=8)


def _get_vocab_size(grammar_obj) -> int:
    if hasattr(grammar_obj, "vocab_size"):
        return grammar_obj.vocab_size
    if hasattr(grammar_obj, "grammar"):
        return _get_vocab_size(grammar_obj.grammar)
    raise AttributeError(
        f"Cannot find vocab_size on grammar object {type(grammar_obj).__name__}"
    )


def batch_apply_grammar_constraints(grammar_objs: list, logits: torch.Tensor) -> None:
    batch_size = logits.shape[0]
    logits_width = logits.shape[1]

    active: List[Tuple[int, object]] = []
    first_grammar = None
    for i, g in enumerate(grammar_objs):
        if g is not None and not g.is_terminated() and not g.finished:
            active.append((i, g))
            if first_grammar is None:
                first_grammar = g

    if not active:
        return

    vocab_size = _get_vocab_size(first_grammar)

    logger.debug(
        "[xgrammar batch_apply] batch=%d, active=%d, logits_width=%d, "
        "grammar_vocab_size=%d, device=%s",
        batch_size,
        len(active),
        logits_width,
        vocab_size,
        logits.device,
    )

    bitmask = allocate_token_bitmask(batch_size, vocab_size)

    if len(active) >= _FILL_PARALLEL_THRESHOLD:
        logger.debug("[xgrammar batch_apply] parallel fill: %d grammars", len(active))
        futures = [_fill_pool.submit(g.fill_vocab_mask, bitmask, i) for i, g in active]
        for f in futures:
            f.result()
    else:
        for i, g in active:
            logger.debug(
                "[xgrammar batch_apply] fill row=%d, type=%s",
                i,
                type(g).__name__,
            )
            g.fill_vocab_mask(bitmask, i)

    bitmask_gpu = bitmask.to(logits.device, non_blocking=True)

    target_logits = logits[:, :vocab_size] if logits_width > vocab_size else logits

    logger.debug("[xgrammar batch_apply] applying triton kernel...")
    apply_token_bitmask_inplace_triton(target_logits, bitmask_gpu)
    logger.debug("[xgrammar batch_apply] done")


def batch_accept_tokens(
    token_grammar_triples: list,
) -> List[Tuple[int, str]]:
    errors: List[Tuple[int, str]] = []
    for idx, triple in enumerate(token_grammar_triples):
        grammar = triple[0]
        token_id = triple[1]
        is_stream_done = triple[2]
        try:
            grammar.accept_token(token_id)
            if is_stream_done:
                grammar.finished = True
                grammar.recycle()
        except Exception as e:
            errors.append((idx, str(e)))
    return errors


def batch_apply_spec_grammar_constraints(
    stream_grammars: list,
    logits: torch.Tensor,
    score_len: int,
) -> None:
    """DFS accept/rollback bitmask generation for speculative decoding.

    Args:
        stream_grammars: list of (grammar_obj_or_None, draft_token_ids_list) per stream.
            draft_token_ids_list has score_len-1 elements (the draft tokens to DFS with).
        logits: [batch_size * score_len, vocab_size] tensor on CUDA.
        score_len: propose_step + 1.
    """
    batch_size = logits.shape[0] // score_len
    logits_width = logits.shape[1]

    active: List[Tuple[int, object, list]] = []
    first_grammar = None
    for i, (grammar, draft_tokens) in enumerate(stream_grammars):
        if grammar is not None and not grammar.is_terminated() and not grammar.finished:
            active.append((i, grammar, draft_tokens))
            if first_grammar is None:
                first_grammar = grammar

    if not active:
        return

    vocab_size = _get_vocab_size(first_grammar)

    logger.debug(
        "[xgrammar spec_apply] batch=%d, score_len=%d, active=%d, "
        "logits_width=%d, vocab_size=%d",
        batch_size,
        score_len,
        len(active),
        logits_width,
        vocab_size,
    )

    # Reuse a single host bitmask buffer across every (stream, position) in
    # this call. xgrammar's fill_next_token_bitmask overwrites the row
    # in-place, so we just need stable host-side storage; the GPU copy
    # produces a fresh device tensor per iteration.
    bitmask = allocate_token_bitmask(1, vocab_size)

    for batch_idx, grammar, draft_tokens in active:
        num_accepted = 0
        try:
            for pos in range(score_len):
                logit_row = batch_idx * score_len + pos

                grammar.fill_vocab_mask(bitmask, 0)
                bitmask_gpu = bitmask.to(logits.device, non_blocking=True)

                target_row = logits[logit_row : logit_row + 1, :vocab_size]
                apply_token_bitmask_inplace_triton(target_row, bitmask_gpu)

                if pos < score_len - 1:
                    grammar.accept_token(draft_tokens[pos])
                    num_accepted += 1

            grammar.rollback(num_accepted)
            logger.debug(
                "[xgrammar spec_apply] stream=%d done: rollback=%d",
                batch_idx,
                num_accepted,
            )
        except Exception as e:
            if num_accepted > 0:
                try:
                    grammar.rollback(num_accepted)
                except Exception:
                    pass
            logger.warning(
                "[xgrammar spec_apply] stream=%d error at pos=%d: %s",
                batch_idx,
                num_accepted,
                e,
            )


def batch_accept_spec_tokens(
    token_grammar_triples: list,
) -> List[Tuple[int, str]]:
    """Accept multiple tokens per grammar for speculative decoding.

    Args:
        token_grammar_triples: list of (grammar, [token_id_0, ..., token_id_N], is_done).
    """
    errors: List[Tuple[int, str]] = []
    for idx, triple in enumerate(token_grammar_triples):
        grammar = triple[0]
        token_ids = triple[1]
        is_stream_done = triple[2]
        try:
            for token_id in token_ids:
                grammar.accept_token(token_id)
            if is_stream_done:
                grammar.finished = True
                grammar.recycle()
        except Exception as e:
            errors.append((idx, str(e)))
    return errors
