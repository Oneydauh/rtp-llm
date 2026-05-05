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

# Opt-in instrumentation for verifying that grammar constraints actually mask
# logits during decode. Enabled per-grammar via GrammarConfig.debug_log.
_DEBUG_MAX_STEPS = 12  # cap per-stream log volume
_debug_seen_steps: dict = {}  # grammar-obj-id -> step count


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

    if getattr(first_grammar, "debug_log", False):
        _dump_bitmask_stats(active, bitmask, vocab_size, logits_width, target_logits)

    logger.debug("[xgrammar batch_apply] applying triton kernel...")
    apply_token_bitmask_inplace_triton(target_logits, bitmask_gpu)
    logger.debug("[xgrammar batch_apply] done")


def _dump_bitmask_stats(active, bitmask, vocab_size, logits_width, target_logits):
    # bitmask is [batch, ceil(vocab_size/32)] int32 packed bits. 1-bit = allowed.
    try:
        bm = bitmask
        for row_idx, g in active:
            gid = id(g)
            step = _debug_seen_steps.get(gid, 0)
            if step >= _DEBUG_MAX_STEPS:
                continue
            _debug_seen_steps[gid] = step + 1

            row = bm[row_idx] if bm.ndim == 2 else bm
            row_int = row.to(torch.int64)
            pop = 0
            allowed_ids: List[int] = []
            for word_idx in range(row_int.numel()):
                w = int(row_int[word_idx].item()) & 0xFFFFFFFF
                if w == 0:
                    continue
                base = word_idx * 32
                for bit in range(32):
                    if w & (1 << bit):
                        pop += 1
                        if len(allowed_ids) < 20:
                            allowed_ids.append(base + bit)
            tok_preview = _decode_tokens(g, allowed_ids)
            # logits slice health-check: are masked positions actually getting
            # pushed to -inf by the triton kernel? Sample the first-row max of a
            # few disallowed positions BEFORE the kernel runs. Cheap.
            logger.info(
                "[xgrammar batch_apply STEP] step=%d row=%d type=%s pattern=%s "
                "n_accepted=%d terminated=%s vocab=%d logits_width=%d "
                "allowed_pop=%d sample_allowed_ids=%s sample_tokens=%s",
                step,
                row_idx,
                type(g).__name__,
                _short_pattern(g),
                len(getattr(g, "accepted_tokens", []) or []),
                getattr(g, "is_terminated", lambda: None)(),
                vocab_size,
                logits_width,
                pop,
                allowed_ids,
                tok_preview,
            )
    except Exception as e:  # pragma: no cover — instrumentation must never crash path
        logger.warning("[xgrammar batch_apply STEP] dump failed: %s", e)


def _short_pattern(g) -> str:
    s = getattr(g, "key_string", None)
    if s is None:
        inner = getattr(g, "grammar", None)
        if inner is not None:
            s = getattr(inner, "key_string", None)
    if not s:
        return "?"
    return s[:80]


def _decode_tokens(g, token_ids) -> list:
    inner = getattr(g, "grammar", g)
    tokenizer = None
    ctx = getattr(inner, "ctx", None)
    if ctx is not None:
        tokenizer_info = getattr(ctx, "tokenizer_info", None)
        if tokenizer_info is not None:
            # xgrammar exposes decoded vocab via tokenizer_info.decoded_vocab
            vocab = getattr(tokenizer_info, "decoded_vocab", None)
            if vocab is not None:
                try:
                    return [vocab[i] if 0 <= i < len(vocab) else "<oob>" for i in token_ids[:8]]
                except Exception:
                    pass
    return []


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
        chain_broken = False
        try:
            for pos in range(score_len):
                logit_row = batch_idx * score_len + pos

                grammar.fill_vocab_mask(bitmask, 0)
                bitmask_gpu = bitmask.to(logits.device, non_blocking=True)

                target_row = logits[logit_row : logit_row + 1, :vocab_size]
                apply_token_bitmask_inplace_triton(target_row, bitmask_gpu)

                if pos < score_len - 1 and not chain_broken:
                    tok = int(draft_tokens[pos])
                    # sglang-style precondition: only accept a draft token into
                    # the matcher if it is legal under the CURRENT mask. Without
                    # this check, xgrammar silently fails accept_token on illegal
                    # tokens (returns False) but num_accepted still increments,
                    # leaving matcher state desynced from num_accepted so the
                    # next position gets a mask for the wrong state.
                    #
                    # When a draft token is illegal, we must NOT break out of
                    # the mask loop — remaining positions still need a mask
                    # based on the current matcher state, or target verify will
                    # sample freely there and emit a grammar-illegal bonus
                    # token. Instead we flag chain_broken and keep filling
                    # masks (all reflecting the frozen matcher state) while
                    # suppressing further accept_token calls.
                    if tok < 0 or tok >= vocab_size:
                        chain_broken = True
                    elif (int(bitmask[0, tok // 32].item()) & (1 << (tok % 32))) == 0:
                        chain_broken = True
                    else:
                        grammar.accept_token(tok)
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


def batch_apply_draft_grammar_constraints(
    stream_grammars: list,
    logits: torch.Tensor,
    step_idx: int,
) -> None:
    """Mask draft logits at a single position along the chain.

    stream_grammars is [(grammar_obj_or_None, draft_tokens_so_far), ...] where
    draft_tokens_so_far[0] is the target-verified bonus token T0 and positions
    [1..] are the draft tokens sampled earlier in this chain. Matcher state at
    entry is "after T0 accept" (prefill bonus accept has already run), so we
    walk draft_tokens_so_far[1:] with accept_token to reach the current chain
    position, fill+apply the bitmask, then rollback exactly what we accepted —
    leaving the matcher in its entry state. applySpecGrammarConstraints (run
    later against the target verify logits) performs its own DFS from the same
    entry state, unaffected.
    """
    active: List[Tuple[int, object, list]] = []
    first_grammar = None
    for i, (grammar, tokens_so_far) in enumerate(stream_grammars):
        if grammar is not None and not grammar.is_terminated() and not grammar.finished:
            active.append((i, grammar, list(tokens_so_far)))
            if first_grammar is None:
                first_grammar = grammar

    if not active:
        return

    vocab_size = _get_vocab_size(first_grammar)
    logits_width = logits.shape[1]
    bitmask = allocate_token_bitmask(1, vocab_size)

    for batch_idx, grammar, tokens_so_far in active:
        to_advance = tokens_so_far[1:]  # skip T0 — already accepted by prefill
        num_accepted = 0
        try:
            # Advance matcher through the already-sampled chain, checking each
            # token against the live mask BEFORE calling accept_token. Without
            # this guard, an illegal token silently fails inside xgrammar but
            # num_accepted still increments, leaving rollback off-by-N and
            # matcher state desynced. If an illegal token shows up, we freeze
            # the matcher and still fill+apply the mask at the paused state
            # so the draft sampler at this step can't pick another illegal
            # continuation on top.
            for tok in to_advance:
                tok_int = int(tok)
                if tok_int < 0 or tok_int >= vocab_size:
                    break
                grammar.fill_vocab_mask(bitmask, 0)
                if (int(bitmask[0, tok_int // 32].item()) & (1 << (tok_int % 32))) == 0:
                    break
                grammar.accept_token(tok_int)
                num_accepted += 1

            grammar.fill_vocab_mask(bitmask, 0)
            bitmask_gpu = bitmask.to(logits.device, non_blocking=True)

            target_row = logits[batch_idx : batch_idx + 1, :vocab_size]
            apply_token_bitmask_inplace_triton(target_row, bitmask_gpu)
        except Exception as e:
            logger.warning(
                "[xgrammar draft_apply] stream=%d step=%d err=%s",
                batch_idx,
                step_idx,
                e,
            )
        finally:
            if num_accepted > 0:
                try:
                    grammar.rollback(num_accepted)
                except Exception as e:
                    logger.warning(
                        "[xgrammar draft_apply] rollback failed stream=%d: %s",
                        batch_idx,
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
