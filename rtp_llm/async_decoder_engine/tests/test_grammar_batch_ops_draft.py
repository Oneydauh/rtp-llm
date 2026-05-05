"""Tests for batch_apply_draft_grammar_constraints (Solution B).

The draft-side mask is supposed to:
  1. Leave matcher state EXACTLY as it found it (accept + rollback balanced)
  2. Actually mask the logits (non-active streams left alone)
  3. Skip terminated / finished / None-grammar streams
  4. Never leak matcher advances across calls
"""

import unittest
from typing import List
from unittest.mock import MagicMock

import torch

from rtp_llm.async_decoder_engine.grammar_batch_ops import (
    batch_apply_draft_grammar_constraints,
)

_CUDA_AVAILABLE = torch.cuda.is_available()


class _MatcherSpy:
    """Minimal fake grammar that records accept/rollback calls for invariant checks."""

    def __init__(self, vocab_size: int = 64, terminated: bool = False):
        self.vocab_size = vocab_size
        self._terminated = terminated
        self.finished = False
        self.accepted: List[int] = []
        self.rollback_counts: List[int] = []
        self._accept_fail_tok = None
        # xgrammar-style; tolerate .matcher access
        self.matcher = MagicMock()

    def is_terminated(self) -> bool:
        return self._terminated

    def accept_token(self, tok: int) -> None:
        if tok == self._accept_fail_tok:
            raise RuntimeError(f"simulated accept failure on token {tok}")
        self.accepted.append(tok)

    def rollback(self, k: int) -> None:
        self.rollback_counts.append(k)
        # keep the accepted list consistent so assertions reflect net state
        del self.accepted[-k:]

    def fill_vocab_mask(self, bitmask: torch.Tensor, idx: int) -> None:
        # Claim only the first 4 token ids. Any offender downstream proves the
        # kernel didn't see this bitmask.
        row = bitmask[idx] if bitmask.ndim == 2 else bitmask
        row.zero_()
        if self.vocab_size >= 4:
            # set bits 0..3
            row[0] = 0xF


class _NoSpyFinished(_MatcherSpy):
    def __init__(self, vocab_size=64):
        super().__init__(vocab_size=vocab_size)
        self.finished = True


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for apply_token_bitmask_inplace_triton")
class TestDraftMaskMatcherInvariant(unittest.TestCase):
    """The core contract: batch_apply_draft_grammar_constraints must leave
    matcher state untouched after return — equal numbers of accept_token and
    rollback calls, with rollback size matching accepts."""

    def _logits(self, batch, vocab):
        return torch.zeros(batch, vocab, dtype=torch.float32, device="cuda")

    def test_empty_active_noop(self):
        g_none = None
        g_term = _MatcherSpy(terminated=True)
        g_fin = _NoSpyFinished()

        logits = self._logits(3, 64)
        stream_grammars = [
            (g_none, []),
            (g_term, [1, 2]),
            (g_fin, [3]),
        ]
        # Should early-return; no mutation to logits (still all zeros) and no
        # matcher calls.
        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=0)

        self.assertTrue(torch.all(logits == 0))
        self.assertEqual(g_term.accepted, [])
        self.assertEqual(g_term.rollback_counts, [])
        self.assertEqual(g_fin.accepted, [])
        self.assertEqual(g_fin.rollback_counts, [])

    def test_single_stream_balanced_accept_rollback(self):
        """1 active stream, draft_tokens_so_far=[T0, d0, d1]. Helper should
        skip T0 and accept d0, d1 → then rollback exactly 2.

        d0, d1 must be inside the fake grammar's mask (bits 0..3) so the
        pre-accept bitmask check lets them through. The outer invariant —
        equal accept+rollback counts — is what this test guards."""
        g = _MatcherSpy()
        logits = self._logits(1, 64)
        stream_grammars = [(g, [999, 1, 2])]  # 999 is T0 (skipped)

        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=1)

        # Net matcher state unchanged.
        self.assertEqual(
            g.accepted, [],
            msg=f"matcher accepted list not balanced: {g.accepted}",
        )
        # Exactly one rollback call of size 2.
        self.assertEqual(g.rollback_counts, [2])

    def test_just_T0_no_accept_no_rollback(self):
        """tokens_so_far = [T0] only → nothing to accept, rollback must be 0
        (or skipped). The invariant is: net accept count = 0, and rollback is
        called only when something was accepted."""
        g = _MatcherSpy()
        logits = self._logits(1, 64)
        stream_grammars = [(g, [42])]

        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=0)

        self.assertEqual(g.accepted, [])
        self.assertEqual(g.rollback_counts, [])  # no-op rollback guard

    def test_multi_stream_isolation(self):
        """3 streams, 2 active. Active matchers each see their own draft chain;
        non-active matcher (None or terminated) never gets accept/rollback."""
        g0 = _MatcherSpy()
        g1 = _MatcherSpy(terminated=True)  # skipped
        g2 = _MatcherSpy()

        logits = self._logits(3, 64)
        stream_grammars = [
            (g0, [11, 1]),       # 1 accept (1), 1 rollback
            (g1, [99, 88, 77]),  # skipped (terminated)
            (g2, [33, 1, 2]),    # 2 accepts, 1 rollback of 2
        ]

        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=2)

        self.assertEqual(g0.accepted, [])
        self.assertEqual(g0.rollback_counts, [1])
        self.assertEqual(g1.accepted, [])
        self.assertEqual(g1.rollback_counts, [])
        self.assertEqual(g2.accepted, [])
        self.assertEqual(g2.rollback_counts, [2])

    def test_accept_failure_still_rollbacks(self):
        """If accept_token throws mid-chain, the finally clause must still
        rollback what was accepted so far — else matcher leaks state.

        All tokens below must pass the fake mask (bits 0..3) so the
        pre-accept bitmask check lets them through; we want the raise to
        happen inside accept_token itself, not be short-circuited by the
        mask guard."""
        g = _MatcherSpy()
        g._accept_fail_tok = 3  # accept of token 3 raises

        logits = self._logits(1, 64)
        # tokens_so_far = [T0, d0, d1_FAIL]. Helper will accept d0=2 (OK),
        # then accept d1=3 which throws. Already-accepted count is 1.
        stream_grammars = [(g, [1, 2, 3])]

        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=2)

        self.assertEqual(g.accepted, [], msg="leaked accept after exception")
        self.assertEqual(g.rollback_counts, [1])


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA required for triton kernel")
class TestDraftMaskLogitsMutation(unittest.TestCase):
    """The helper must actually push disallowed logits to -inf for active
    streams, and leave non-active streams untouched."""

    def test_only_active_row_is_masked(self):
        g0 = _MatcherSpy()   # active
        g1 = None            # no grammar
        g2 = _MatcherSpy(terminated=True)  # terminated

        logits = torch.zeros(3, 64, dtype=torch.float32, device="cuda")
        stream_grammars = [(g0, [1, 2]), (g1, []), (g2, [3])]

        batch_apply_draft_grammar_constraints(stream_grammars, logits, step_idx=1)

        # Row 0: bits 0..3 allowed → positions 0..3 are finite, 4..63 are -inf.
        row0 = logits[0]
        self.assertTrue(torch.all(torch.isfinite(row0[:4])))
        self.assertTrue(torch.all(torch.isinf(row0[4:])))

        # Row 1 and 2: untouched, still all zeros.
        self.assertTrue(torch.all(logits[1] == 0), msg=f"row 1 leaked: {logits[1]}")
        self.assertTrue(torch.all(logits[2] == 0), msg=f"row 2 leaked: {logits[2]}")


class TestDraftMaskNoCuda(unittest.TestCase):
    """The non-CUDA path should not crash when given an empty active set — a
    smoke-level guard for CPU-only dev environments."""

    def test_all_none_stream_grammars(self):
        # Pure no-op; no CUDA tensor needed because we never enter the active
        # branch. Use a CPU tensor to confirm the early-return.
        logits = torch.zeros(2, 64, dtype=torch.float32)
        batch_apply_draft_grammar_constraints([(None, []), (None, [])], logits, 0)
        self.assertTrue(torch.all(logits == 0))


if __name__ == "__main__":
    unittest.main()
