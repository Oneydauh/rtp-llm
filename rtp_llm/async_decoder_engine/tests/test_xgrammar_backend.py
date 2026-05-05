"""Tests for xgrammar_backend fixes: imports, reset, apply_vocab_mask, response_format."""

import json
import unittest
from unittest.mock import MagicMock, patch


class TestImports(unittest.TestCase):
    """Verify all required symbols are importable."""

    def test_grammar_stats_imported_from_base(self):
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            GrammarStats as BaseGrammarStats,
        )
        from rtp_llm.async_decoder_engine.xgrammar_backend import GrammarStats

        self.assertIs(GrammarStats, BaseGrammarStats)

    def test_structural_tag_importable(self):
        try:
            from rtp_llm.async_decoder_engine.xgrammar_backend import (  # noqa: F401
                StructuralTag,
            )
        except ImportError:
            self.fail(
                "StructuralTag should be importable via xgrammar_backend module scope"
            )

    def test_triton_bitmask_importable(self):
        from rtp_llm.models_py.triton_kernels.grammar.bitmask_ops import (  # noqa: F401
            apply_token_bitmask_inplace_triton,
        )


class TestXGrammarGrammarApplyVocabMask(unittest.TestCase):
    """Test that apply_vocab_mask dispatches correctly."""

    def test_cpu_raises(self):
        import torch

        from rtp_llm.async_decoder_engine.xgrammar_backend import XGrammarGrammar

        mock_matcher = MagicMock()
        mock_matcher.is_terminated.return_value = False

        grammar = XGrammarGrammar(
            matcher=mock_matcher,
            vocab_size=100,
            ctx=MagicMock(),
            override_stop_tokens=None,
        )

        logits = torch.ones(1, 100, dtype=torch.float32)
        mask = torch.ones(1, 4, dtype=torch.int32)
        with self.assertRaises(RuntimeError):
            grammar.apply_vocab_mask(logits, mask)

    @unittest.skipUnless(__import__("torch").cuda.is_available(), "CUDA not available")
    def test_cuda_dispatches_triton(self):
        import torch

        from rtp_llm.async_decoder_engine.xgrammar_backend import XGrammarGrammar

        mock_matcher = MagicMock()
        mock_matcher.is_terminated.return_value = False

        grammar = XGrammarGrammar(
            matcher=mock_matcher,
            vocab_size=64,
            ctx=MagicMock(),
            override_stop_tokens=None,
        )

        logits = torch.ones(1, 64, dtype=torch.float32, device="cuda")
        bitmask = torch.zeros(1, 2, dtype=torch.int32, device="cuda")

        grammar.apply_vocab_mask(logits, bitmask)
        self.assertTrue(torch.all(torch.isinf(logits)))


class TestXGrammarGrammarBackendReset(unittest.TestCase):
    """Verify reset() clears both compiler cache and base dict cache."""

    def test_reset_clears_base_cache(self):
        from rtp_llm.async_decoder_engine.xgrammar_backend import XGrammarGrammarBackend

        # Use a plain object without `init_xgrammar` so the backend falls
        # through to the TokenizerInfo.from_huggingface path (mocked below).
        # MagicMock auto-creates attrs, which makes `hasattr(mock, "init_xgrammar")`
        # return True and unpacks an auto-generated MagicMock as a 2-tuple — that
        # raises ValueError. Keep the tokenizer attr-free instead.
        class _FakeTokenizer:
            pass

        mock_tokenizer = _FakeTokenizer()
        mock_tokenizer_info = MagicMock()
        mock_compiler = MagicMock()

        with patch(
            "rtp_llm.async_decoder_engine.xgrammar_backend.TokenizerInfo"
        ) as MockTI, patch(
            "rtp_llm.async_decoder_engine.xgrammar_backend.GrammarCompiler"
        ) as MockGC:
            MockTI.from_huggingface.return_value = mock_tokenizer_info
            MockGC.return_value = mock_compiler

            backend = XGrammarGrammarBackend(tokenizer=mock_tokenizer, vocab_size=100)

        backend.cache[("json", '{"type":"object"}')] = MagicMock()
        self.assertEqual(len(backend.cache), 1)

        backend.reset()

        self.assertEqual(len(backend.cache), 0)
        mock_compiler.clear_cache.assert_called_once()


class TestApplyResponseFormat(unittest.TestCase):
    """Verify _apply_response_format serializes json_schema as string."""

    def test_json_schema_serialized_to_string(self):
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint

        class FakeSchemaObj:
            schema = {"type": "object", "properties": {"name": {"type": "string"}}}

        class FakeRF:
            type = "json_schema"
            json_schema = FakeSchemaObj()
            pattern = None
            grammar = None
            structural_tag = None

        config = GenerateConfig()
        OpenaiEndpoint._apply_response_format(FakeRF(), config)

        self.assertIsInstance(config.json_schema, str)
        parsed = json.loads(config.json_schema)
        self.assertEqual(parsed["type"], "object")
        self.assertIn("properties", parsed)

    def test_json_object_serialized_to_string(self):
        from rtp_llm.config.generate_config import GenerateConfig
        from rtp_llm.openai.openai_endpoint import OpenaiEndpoint

        class FakeRF:
            type = "json_object"
            json_schema = None
            pattern = None
            grammar = None
            structural_tag = None

        config = GenerateConfig()
        OpenaiEndpoint._apply_response_format(FakeRF(), config)

        self.assertIsInstance(config.json_schema, str)
        parsed = json.loads(config.json_schema)
        self.assertEqual(parsed, {"type": "object"})


class TestBitmaskDirectAssertion(unittest.TestCase):
    """Direct-inspection assertions on the xgrammar bitmask at known matcher
    states. This is what guarantees xgrammar is masking *correctly*, not
    just that the end-to-end output happens to match a pattern.

    The other grammar smokes check: "final output fullmatches pattern X" —
    that is indirect evidence. A bitmask that erroneously ALLOWS extra
    tokens can still pass those checks if greedy temp=0 sampling never
    picks the over-allowed tokens.

    These tests open the black box: they compile a regex grammar, call the
    same batch_apply_grammar_constraints used on the hot path, then
    inspect which logit positions survived and decode them via the
    tokenizer to assert every single allowed token is consistent with the
    regex state.
    """

    QWEN2_MODEL_PATH = "/home/yanxi.wln/work/models/Qwen2-1.5B-Instruct/"

    @classmethod
    def setUpClass(cls):
        import os

        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for triton bitmask kernel")
        if not os.path.isdir(cls.QWEN2_MODEL_PATH):
            raise unittest.SkipTest(
                f"tokenizer path not found: {cls.QWEN2_MODEL_PATH}"
            )
        try:
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained(cls.QWEN2_MODEL_PATH)
        except Exception as e:
            raise unittest.SkipTest(f"failed to load tokenizer: {e}")

        from rtp_llm.async_decoder_engine.xgrammar_backend import (
            XGrammarGrammarBackend,
        )

        cls.backend = XGrammarGrammarBackend(
            tokenizer=cls.tokenizer,
            vocab_size=len(cls.tokenizer),
        )

    def _fresh_grammar_at_start(self, regex_pattern: str):
        """Compile a regex grammar and return a matcher at state START."""
        grammar = self.backend.dispatch_regex(regex_pattern)
        # Sanity: we got a real grammar, not an InvalidGrammarObject.
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        self.assertNotIsInstance(
            grammar,
            InvalidGrammarObject,
            msg=f"regex failed to compile: {regex_pattern}",
        )
        return grammar

    def _allowed_token_ids(self, grammar, vocab_size: int):
        """Run the hot-path bitmask apply and return the ids that survived."""
        import torch

        from rtp_llm.async_decoder_engine.grammar_batch_ops import (
            batch_apply_grammar_constraints,
        )

        # Use zero logits so that whatever survives is purely the bitmask's
        # contribution. The triton kernel pushes disallowed positions to -inf.
        logits = torch.zeros(1, vocab_size, dtype=torch.float32, device="cuda")
        batch_apply_grammar_constraints([grammar], logits)
        allowed_mask = torch.isfinite(logits[0])
        return torch.nonzero(allowed_mask, as_tuple=True)[0].cpu().tolist()

    def test_hex_color_regex_at_start_only_hash_prefixed(self):
        """For regex `#[0-9A-Fa-f]{6}` at state START, every allowed token
        must decode to a string beginning with `#` (and containing no
        characters outside `#[0-9A-Fa-f]`). A leaky bitmask that also
        allowed, say, alphabetic tokens would fail this check immediately.
        """
        pattern = "#[0-9A-Fa-f]{6}"
        grammar = self._fresh_grammar_at_start(pattern)
        vocab_size = grammar.vocab_size

        allowed_ids = self._allowed_token_ids(grammar, vocab_size)

        self.assertGreater(
            len(allowed_ids), 0, "bitmask allowed zero tokens at state START"
        )
        # `#` is a rare prefix in Qwen2's vocab; we expect under 200 such
        # tokens. Use a generous cap so this doesn't become flaky against
        # tokenizer revisions — the point is "not thousands".
        self.assertLess(
            len(allowed_ids),
            300,
            f"bitmask allowed {len(allowed_ids)} tokens at START for "
            f"pattern {pattern!r}; expected << vocab_size (~{vocab_size})",
        )

        import re as _re

        hex_only = _re.compile(r"^#[0-9A-Fa-f]*$")
        offenders = []
        for tid in allowed_ids:
            s = self.tokenizer.decode([tid])
            if not hex_only.match(s):
                offenders.append((tid, s))
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"bitmask allowed tokens incompatible with pattern {pattern!r} "
                f"at state START. Offenders (up to 10 shown): "
                f"{offenders[:10]}"
            ),
        )

    def test_zip_regex_at_start_only_digits(self):
        """For regex `[0-9]{5}` at state START, every allowed token must
        decode to a string containing only ASCII digits 0-9 (allowing the
        empty string for BOS/special tokens that xgrammar may treat as
        pass-through)."""
        pattern = "[0-9]{5}"
        grammar = self._fresh_grammar_at_start(pattern)
        vocab_size = grammar.vocab_size

        allowed_ids = self._allowed_token_ids(grammar, vocab_size)
        self.assertGreater(len(allowed_ids), 0)

        import re as _re

        digits_only = _re.compile(r"^[0-9]*$")
        offenders = []
        for tid in allowed_ids:
            s = self.tokenizer.decode([tid])
            if not digits_only.match(s):
                offenders.append((tid, s))
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"bitmask allowed non-digit tokens at state START for "
                f"pattern {pattern!r}. Offenders (up to 10 shown): "
                f"{offenders[:10]}"
            ),
        )

    def test_matcher_advance_changes_bitmask(self):
        """Sanity: state-transition changes the bitmask. After
        accept_token on a valid start token for `#[0-9A-Fa-f]{6}`, the
        new allowed set must be a DIFFERENT set (hex digits only, no
        more `#`-prefixed strings). This catches matcher advance bugs
        that would leave state frozen at START.
        """
        pattern = "#[0-9A-Fa-f]{6}"
        grammar = self._fresh_grammar_at_start(pattern)
        vocab_size = grammar.vocab_size

        start_allowed = set(self._allowed_token_ids(grammar, vocab_size))

        # Pick a token that decodes to "#" and nothing else — advance.
        hash_tokens = self.tokenizer.encode("#", add_special_tokens=False)
        self.assertTrue(
            len(hash_tokens) >= 1,
            "tokenizer could not produce a '#' token for advance test",
        )
        grammar.accept_token(hash_tokens[0])

        after_allowed = set(self._allowed_token_ids(grammar, vocab_size))

        self.assertNotEqual(
            start_allowed,
            after_allowed,
            msg=(
                "matcher state did not change after accept_token; "
                "bitmask at 'after #' is identical to bitmask at START — "
                "this is the ##a3e3f9-style bug"
            ),
        )

        # And: any token now allowed must be a valid continuation (hex
        # digits only, no more '#').
        import re as _re

        hex_cont = _re.compile(r"^[0-9A-Fa-f]*$")
        offenders = []
        for tid in after_allowed:
            s = self.tokenizer.decode([tid])
            if not hex_cont.match(s):
                offenders.append((tid, s))
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"after accept_token('#'), bitmask still allows non-hex "
                f"continuation tokens. Offenders (up to 10): "
                f"{offenders[:10]}"
            ),
        )


class TestBitmaskMidSequenceAssertion(unittest.TestCase):
    """Bitmask assertions at states DEEP inside the grammar, not just START.

    The earlier TestBitmaskDirectAssertion only covers state=START and one
    accept step. That leaves the largest invariant blind-spot: if the mask
    is correct at the boundary but desyncs mid-sequence (e.g. the NFA
    transition table is traversed with an off-by-one, or rollback restores
    the wrong state), START-only tests won't notice — the end-to-end
    output can still `fullmatch` when the bug is narrow enough to let
    greedy sampling limp through.

    These tests drive the matcher through real grammar transitions and
    assert the allowed set at each mid-state against what the pattern
    logically requires.
    """

    QWEN2_MODEL_PATH = "/home/yanxi.wln/work/models/Qwen2-1.5B-Instruct/"

    @classmethod
    def setUpClass(cls):
        import os

        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required for triton bitmask kernel")
        if not os.path.isdir(cls.QWEN2_MODEL_PATH):
            raise unittest.SkipTest(
                f"tokenizer path not found: {cls.QWEN2_MODEL_PATH}"
            )
        try:
            from transformers import AutoTokenizer

            cls.tokenizer = AutoTokenizer.from_pretrained(cls.QWEN2_MODEL_PATH)
        except Exception as e:
            raise unittest.SkipTest(f"failed to load tokenizer: {e}")

        from rtp_llm.async_decoder_engine.xgrammar_backend import (
            XGrammarGrammarBackend,
        )

        cls.backend = XGrammarGrammarBackend(
            tokenizer=cls.tokenizer,
            vocab_size=len(cls.tokenizer),
        )

    def _grammar_regex(self, pattern: str):
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        g = self.backend.dispatch_regex(pattern)
        self.assertNotIsInstance(g, InvalidGrammarObject)
        return g

    def _grammar_json(self, schema_str: str):
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        g = self.backend.dispatch_json(schema_str)
        self.assertNotIsInstance(g, InvalidGrammarObject)
        return g

    def _allowed_ids(self, grammar):
        """Return the list of vocab ids allowed by the current matcher state."""
        import torch

        from rtp_llm.async_decoder_engine.grammar_batch_ops import (
            batch_apply_grammar_constraints,
        )

        # Make a fresh logits tensor each call — the kernel writes -inf into
        # it in place.
        logits = torch.zeros(
            1, grammar.vocab_size, dtype=torch.float32, device="cuda"
        )
        batch_apply_grammar_constraints([grammar], logits)
        return torch.nonzero(torch.isfinite(logits[0]), as_tuple=True)[0].cpu().tolist()

    def _single_token_for(self, s: str) -> int:
        """Return a token id whose `.decode()` is exactly `s`, or skip the
        test if the tokenizer refuses to produce one. Needed because we
        must drive accept_token with concrete integer ids."""
        tokens = self.tokenizer.encode(s, add_special_tokens=False)
        # Prefer a single-token encoding if available.
        for tid in tokens:
            if self.tokenizer.decode([tid]) == s:
                return tid
        raise unittest.SkipTest(
            f"tokenizer has no single-token encoding for {s!r}; "
            f"got tokens={tokens}, decoded pieces="
            f"{[self.tokenizer.decode([t]) for t in tokens]}"
        )

    # -- mid-sequence tests -------------------------------------------------

    def test_after_one_hex_only_hex_continuation(self):
        """After `#` + 1 hex digit, bitmask must still allow ONLY hex
        continuations (5 left), never `#` again, never alpha beyond a-f."""
        import re as _re

        pattern = "#[0-9A-Fa-f]{6}"
        grammar = self._grammar_regex(pattern)

        hash_tok = self._single_token_for("#")
        digit_tok = self._single_token_for("a")
        grammar.accept_token(hash_tok)
        grammar.accept_token(digit_tok)

        allowed = self._allowed_ids(grammar)
        self.assertGreater(len(allowed), 0)

        hex_only = _re.compile(r"^[0-9A-Fa-f]*$")
        offenders = [
            (tid, self.tokenizer.decode([tid]))
            for tid in allowed
            if not hex_only.match(self.tokenizer.decode([tid]))
        ]
        self.assertEqual(
            offenders,
            [],
            msg=(
                f"after accept('#','a'), bitmask at 5-hex-left state allows "
                f"non-hex tokens. Offenders: {offenders[:10]}"
            ),
        )
        # Explicit negative: the `#` token must NOT be allowed mid-sequence.
        self.assertNotIn(
            hash_tok,
            allowed,
            msg="'#' token allowed mid-sequence — duplicated-start bug class",
        )

    def test_near_terminal_still_requires_one_hex(self):
        """After `#` + 5 hex digits, grammar still expects exactly 1 more
        hex digit and nothing else. A frozen-state bug would have the
        matcher still allowing tokens that were valid 4 steps ago."""
        import re as _re

        pattern = "#[0-9A-Fa-f]{6}"
        grammar = self._grammar_regex(pattern)

        for ch in "#abcde":
            grammar.accept_token(self._single_token_for(ch))

        allowed = self._allowed_ids(grammar)
        self.assertGreater(len(allowed), 0)

        hex_only = _re.compile(r"^[0-9A-Fa-f]*$")
        # Every allowed token must be purely hex (no `#`, no letters beyond
        # a-f). We cannot easily enforce "exactly one hex char" here because
        # the tokenizer may have multi-char hex tokens — but none of those
        # should extend past the grammar length, and xgrammar is expected
        # to reject the over-long ones. At minimum, non-hex tokens must be
        # fully excluded.
        offenders = [
            (tid, self.tokenizer.decode([tid]))
            for tid in allowed
            if not hex_only.match(self.tokenizer.decode([tid]))
        ]
        self.assertEqual(
            offenders, [], msg=f"near-terminal offenders: {offenders[:10]}"
        )

    def test_terminal_state_allows_termination(self):
        """After accepting a complete match `#AABBCC`, the matcher must be
        in an accepting state where EITHER (a) is_terminated() is True, or
        (b) the allowed set is limited to stop/EOS (no more hex digits —
        the grammar has consumed its full budget). A bitmask bug that
        keeps advancing past the grammar limit is the `#ABCDEF<junk>`
        class of error."""
        pattern = "#[0-9A-Fa-f]{6}"
        grammar = self._grammar_regex(pattern)

        for ch in "#ABCDEF":
            grammar.accept_token(self._single_token_for(ch))

        # After a full regex match, xgrammar should recognise the matcher
        # has reached an accepting end-state. Either flag suffices.
        is_term = grammar.matcher.is_terminated()

        allowed = self._allowed_ids(grammar)

        # If the matcher thinks it's terminated, the bitmask is either empty
        # or restricted — in either case, non-terminal hex digits must not
        # be allowed.
        #
        # Define an "accepts EOS" reference: get EOS token id(s) from the
        # tokenizer and check if any are in `allowed`.
        eos_ids = set()
        try:
            eos = self.tokenizer.eos_token_id
            if eos is not None:
                eos_ids.add(int(eos))
        except Exception:
            pass
        try:
            # Some tokenizers expose additional stop ids via special tokens.
            added = self.tokenizer.added_tokens_encoder
            for tok_str, tok_id in (added or {}).items():
                if "eos" in tok_str.lower() or tok_str in ("<|endoftext|>", "<|im_end|>"):
                    eos_ids.add(int(tok_id))
        except Exception:
            pass

        accepts_eos = any(tid in allowed for tid in eos_ids) or is_term
        self.assertTrue(
            accepts_eos,
            msg=(
                f"after full match #ABCDEF, matcher neither terminated nor "
                f"allows any EOS token (eos_ids={sorted(eos_ids)}, "
                f"allowed_count={len(allowed)}, "
                f"first_allowed_samples="
                f"{[self.tokenizer.decode([t]) for t in allowed[:10]]})"
            ),
        )

        # The allowed set must not contain tokens that would extend past
        # the match — e.g. any hex digit.
        hex_ch_tokens = set()
        for ch in "0123456789abcdefABCDEF":
            try:
                hex_ch_tokens.add(self._single_token_for(ch))
            except unittest.SkipTest:
                continue
        leaky = [tid for tid in allowed if tid in hex_ch_tokens]
        self.assertEqual(
            leaky,
            [],
            msg=(
                f"after full match #ABCDEF, bitmask still allows hex "
                f"continuation tokens — grammar budget over-run: "
                f"{[self.tokenizer.decode([t]) for t in leaky]}"
            ),
        )

    def test_json_integer_field_mid_state_no_alpha(self):
        """Compile a schema with an integer field, drive matcher to the
        point of value emission, and assert alphabetic tokens are masked
        out. A mask that still allows letters at an integer position
        would let the model produce invalid JSON like `{"n": abc}`."""
        import re as _re

        schema = json.dumps(
            {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
                "additionalProperties": False,
            }
        )
        grammar = self._grammar_json(schema)

        # Greedily drive the matcher with single-char tokens for the prefix
        # `{"n":`. Skip if any char lacks a clean single-token encoding
        # (tokenizer-dependent) — we still cover the general assertion when
        # the tokenizer cooperates.
        try:
            prefix_tokens = [self._single_token_for(ch) for ch in '{"n":']
        except unittest.SkipTest:
            # Fall back to encoding the whole prefix as a run of tokens; we
            # don't care about exact segmentation as long as xgrammar
            # accepts each one.
            prefix_tokens = self.tokenizer.encode(
                '{"n":', add_special_tokens=False
            )

        for tid in prefix_tokens:
            try:
                grammar.accept_token(tid)
            except Exception as e:
                raise unittest.SkipTest(
                    f"xgrammar rejected token in prefix (tokenizer/grammar "
                    f"tokenization mismatch): tid={tid} "
                    f"piece={self.tokenizer.decode([tid])!r} err={e}"
                )

        allowed = self._allowed_ids(grammar)
        self.assertGreater(len(allowed), 0)

        # At this state, the next token should start a JSON integer value.
        # Valid starts: optional leading whitespace, then `-` or `[0-9]`.
        # We assert: any allowed token that decodes to an ASCII-letter-only
        # string is a bug — letters cannot start an integer literal.
        alpha_only = _re.compile(r"^[A-Za-z]+$")
        offenders = []
        for tid in allowed:
            s = self.tokenizer.decode([tid])
            if s and alpha_only.match(s):
                offenders.append((tid, s))

        self.assertEqual(
            offenders,
            [],
            msg=(
                f"at JSON integer-value state, bitmask allows pure-alpha "
                f"tokens: {offenders[:10]}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
