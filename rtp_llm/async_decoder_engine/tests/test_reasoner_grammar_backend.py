"""Tests for ReasonerGrammarBackend / ReasonerGrammarObject.

Semantics:
  * tokens_after_think_end == -1 : still reasoning. Grammar is a pass-through:
    accept_token does NOT advance the inner grammar, fill_vocab_mask does
    nothing, rollback touches only the outer state counter.
  * tokens_after_think_end == 0  : we just saw think_end_id. Next token is
    subject to the inner grammar.
  * tokens_after_think_end > 0   : full grammar mode.

The transition is driven by observing the `think_end_id` token via
accept_token. Rollback must undo the transition symmetrically.
"""

import unittest
from typing import List, Tuple
from unittest.mock import MagicMock

from rtp_llm.async_decoder_engine.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)
from rtp_llm.async_decoder_engine.reasoner_grammar_backend import (
    ReasonerGrammarBackend,
    ReasonerGrammarObject,
)


class _InnerGrammar(BaseGrammarObject):
    """Records calls from the reasoner wrapper to prove dispatch correctness."""

    def __init__(self, vocab_size: int = 32):
        super().__init__()
        self.vocab_size = vocab_size
        self.accepted: List[int] = []
        self.rollback_counts: List[int] = []
        self.fill_calls: List[int] = []
        self._terminated = False

    def accept_token(self, token: int):
        self.accepted.append(token)

    def rollback(self, k: int):
        self.rollback_counts.append(k)

    def fill_vocab_mask(self, bitmask, idx: int):
        self.fill_calls.append(idx)

    def is_terminated(self):
        return self._terminated

    def copy(self):
        clone = _InnerGrammar(self.vocab_size)
        clone.accepted = list(self.accepted)
        return clone


class _InnerBackend(BaseGrammarBackend):
    def __init__(self):
        super().__init__()

    def _init_value_dispatch(
        self, key: Tuple[str, str], reasoning: bool
    ) -> BaseGrammarObject:
        if key[1] == "__invalid__":
            return InvalidGrammarObject("forced-invalid")
        return _InnerGrammar()


class TestReasonerObjectStateMachine(unittest.TestCase):
    THINK_END_ID = 7

    def _fresh(self, reasoning: bool = True) -> ReasonerGrammarObject:
        inner = _InnerGrammar()
        obj = ReasonerGrammarObject(inner, self.THINK_END_ID)
        obj.maybe_init_reasoning(reasoning)
        return obj

    # -- during reasoning (tokens_after_think_end == -1) --------------------

    def test_pre_end_accept_does_not_touch_inner(self):
        obj = self._fresh(reasoning=True)
        for t in [1, 2, 3]:
            obj.accept_token(t)
        self.assertEqual(obj.grammar.accepted, [])
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_pre_end_fill_is_passthrough_noop(self):
        obj = self._fresh(reasoning=True)
        obj.fill_vocab_mask(MagicMock(), 0)
        self.assertEqual(obj.grammar.fill_calls, [])

    # -- the transition ------------------------------------------------------

    def test_think_end_flips_state(self):
        obj = self._fresh(reasoning=True)
        obj.accept_token(1)
        obj.accept_token(self.THINK_END_ID)
        # After transfer_state, we've "just ended". Inner still not touched
        # because accept_token gates on the *prior* state.
        self.assertEqual(obj.tokens_after_think_end, 0)
        self.assertEqual(obj.grammar.accepted, [])

    def test_post_end_next_token_feeds_inner(self):
        obj = self._fresh(reasoning=True)
        obj.accept_token(self.THINK_END_ID)  # now state=0
        obj.accept_token(42)                  # state=1, inner accepts 42
        self.assertEqual(obj.grammar.accepted, [42])
        self.assertEqual(obj.tokens_after_think_end, 1)

    def test_reasoning_false_starts_in_grammar_mode(self):
        """maybe_init_reasoning(False) skips the warmup and starts at state=0
        (grammar active from the first token)."""
        obj = self._fresh(reasoning=False)
        self.assertEqual(obj.tokens_after_think_end, 0)
        obj.accept_token(99)
        self.assertEqual(obj.grammar.accepted, [99])

    # -- rollback symmetry ---------------------------------------------------

    def test_rollback_in_grammar_zone_forwards_to_inner(self):
        obj = self._fresh(reasoning=False)
        obj.accept_token(10)
        obj.accept_token(20)
        self.assertEqual(obj.tokens_after_think_end, 2)
        self.assertEqual(obj.grammar.accepted, [10, 20])

        obj.rollback(1)
        # one rollback of size 1 to inner, and outer counter decremented.
        self.assertEqual(obj.grammar.rollback_counts, [1])
        self.assertEqual(obj.tokens_after_think_end, 1)

    def test_rollback_across_think_end_partial_inner(self):
        """Rollback of k that crosses think_end: inner only rolls back the
        tokens that were in the grammar zone, outer steps through all k."""
        obj = self._fresh(reasoning=True)
        obj.accept_token(1)
        obj.accept_token(2)
        obj.accept_token(self.THINK_END_ID)  # state=0
        obj.accept_token(10)                  # state=1
        obj.accept_token(20)                  # state=2
        self.assertEqual(obj.grammar.accepted, [10, 20])

        # Rollback 4 steps: 2 grammar tokens + 1 for "end of think" + 1 pre.
        obj.rollback(4)

        # Inner rollback is min(k, tokens_after_think_end_before_rollback) = 2.
        self.assertEqual(obj.grammar.rollback_counts, [2])
        # Outer state fully unwound back to pre-end.
        self.assertEqual(obj.tokens_after_think_end, -1)

    def test_rollback_purely_in_reasoning_zone_inner_untouched(self):
        obj = self._fresh(reasoning=True)
        obj.accept_token(1)
        obj.accept_token(2)
        obj.rollback(2)
        self.assertEqual(obj.grammar.rollback_counts, [])
        self.assertEqual(obj.tokens_after_think_end, -1)

    # -- passthroughs --------------------------------------------------------

    def test_is_terminated_delegates(self):
        obj = self._fresh(reasoning=False)
        self.assertFalse(obj.is_terminated())
        obj.grammar._terminated = True
        self.assertTrue(obj.is_terminated())

    def test_finished_property_delegates(self):
        obj = self._fresh(reasoning=False)
        self.assertFalse(obj.finished)
        obj.finished = True
        self.assertTrue(obj.grammar.finished)

    def test_copy_preserves_think_end_id(self):
        obj = self._fresh(reasoning=False)
        obj.accept_token(55)
        c = obj.copy()
        self.assertEqual(c.think_end_id, self.THINK_END_ID)
        self.assertIsInstance(c, ReasonerGrammarObject)
        # The inner copy must not share state with the original's inner.
        self.assertIsNot(c.grammar, obj.grammar)


class TestReasonerBackendDispatch(unittest.TestCase):
    def test_dispatch_wraps_with_reasoner_object(self):
        inner_be = _InnerBackend()
        be = ReasonerGrammarBackend(inner_be, think_end_id=7)
        g = be._init_value_dispatch(("json", "{}"), reasoning=True)
        self.assertIsInstance(g, ReasonerGrammarObject)
        self.assertEqual(g.think_end_id, 7)
        self.assertEqual(g.tokens_after_think_end, -1)

    def test_dispatch_reasoning_false_starts_active(self):
        inner_be = _InnerBackend()
        be = ReasonerGrammarBackend(inner_be, think_end_id=7)
        g = be._init_value_dispatch(("json", "{}"), reasoning=False)
        self.assertIsInstance(g, ReasonerGrammarObject)
        self.assertEqual(g.tokens_after_think_end, 0)

    def test_invalid_inner_passes_through_unwrapped(self):
        """When the inner backend produces InvalidGrammarObject, the reasoner
        backend must NOT wrap it — the scheduler inspects the type to reject
        the request. Wrapping would hide the invalidity."""
        inner_be = _InnerBackend()
        be = ReasonerGrammarBackend(inner_be, think_end_id=7)
        g = be._init_value_dispatch(("json", "__invalid__"), reasoning=True)
        self.assertIsInstance(g, InvalidGrammarObject)

    def test_none_inner_passes_through(self):
        """If inner returns None (e.g. disabled backend), reasoner must also
        return None — no wrapping of nothing."""

        class _NoneBackend(BaseGrammarBackend):
            def _init_value_dispatch(self, key, reasoning):
                return None

        be = ReasonerGrammarBackend(_NoneBackend(), think_end_id=7)
        g = be._init_value_dispatch(("json", "{}"), reasoning=True)
        self.assertIsNone(g)


if __name__ == "__main__":
    unittest.main()
