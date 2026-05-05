"""Unit tests for the BaseGrammarBackend synchronous API.

The C++ GrammarManager drives compile_now / get_cached / set_cache directly,
so these three entry points need to preserve their contract across refactors.
This test suite does NOT exercise xgrammar itself — it uses a subclass that
fakes _init_value_dispatch — so it's fast and has no GPU / tokenizer deps.
"""

import threading
import time
import unittest
from typing import Tuple

from rtp_llm.async_decoder_engine.base_grammar_backend import (
    BaseGrammarBackend,
    BaseGrammarObject,
    InvalidGrammarObject,
)


class FakeGrammarObject(BaseGrammarObject):
    """Minimal grammar object that records its own identity for assertions."""

    def __init__(self, tag: str):
        super().__init__()
        self.tag = tag
        self.reasoning_initialized = False

    def copy(self) -> "FakeGrammarObject":
        clone = FakeGrammarObject(self.tag)
        clone.reasoning_initialized = self.reasoning_initialized
        return clone

    def maybe_init_reasoning(self, reasoning: bool) -> None:
        if reasoning:
            self.reasoning_initialized = True


class FakeBackend(BaseGrammarBackend):
    def __init__(self, compile_delay_s: float = 0.0, fail_keys=None):
        super().__init__()
        self.compile_delay_s = compile_delay_s
        self.fail_keys = set(fail_keys or [])
        self.compile_count = 0
        self.compile_lock = threading.Lock()

    def _init_value_dispatch(
        self, key: Tuple[str, str], require_reasoning: bool
    ) -> BaseGrammarObject:
        with self.compile_lock:
            self.compile_count += 1
        if self.compile_delay_s:
            time.sleep(self.compile_delay_s)
        if key in self.fail_keys:
            return InvalidGrammarObject(f"fail: {key}")
        obj = FakeGrammarObject(tag=f"{key[0]}:{key[1]}")
        obj.maybe_init_reasoning(require_reasoning)
        return obj


class TestInvalidGrammarClsExposed(unittest.TestCase):
    def test_class_attr_points_to_invalid_class(self):
        # GrammarManager constructor looks up this attribute by name.
        self.assertIs(BaseGrammarBackend._invalid_grammar_cls, InvalidGrammarObject)


class TestGetCached(unittest.TestCase):
    def test_miss_returns_none(self):
        backend = FakeBackend()
        self.assertIsNone(backend.get_cached(("json", "{}"), require_reasoning=False))

    def test_hit_returns_copy_not_cache_entry(self):
        backend = FakeBackend()
        original = FakeGrammarObject("json:{}")
        backend.set_cache(("json", "{}"), original)

        first = backend.get_cached(("json", "{}"), require_reasoning=False)
        second = backend.get_cached(("json", "{}"), require_reasoning=False)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, original)
        self.assertIsNot(first, second)

    def test_hit_initializes_reasoning_on_copy(self):
        backend = FakeBackend()
        backend.set_cache(("json", "{}"), FakeGrammarObject("json:{}"))

        with_reasoning = backend.get_cached(("json", "{}"), require_reasoning=True)
        without = backend.get_cached(("json", "{}"), require_reasoning=False)

        self.assertTrue(with_reasoning.reasoning_initialized)
        self.assertFalse(without.reasoning_initialized)


class TestCompileNow(unittest.TestCase):
    def test_returns_grammar_and_does_not_touch_cache(self):
        backend = FakeBackend()
        result = backend.compile_now(("json", "{}"), require_reasoning=False)

        self.assertIsInstance(result, FakeGrammarObject)
        # compile_now is synchronous; C++ owns the set_cache call.
        self.assertEqual(backend.cache, {})

    def test_returns_invalid_on_failure(self):
        backend = FakeBackend(fail_keys={("regex", "(")})
        result = backend.compile_now(("regex", "("), require_reasoning=False)
        self.assertIsInstance(result, InvalidGrammarObject)

    def test_concurrent_compile_calls_are_independent(self):
        # Simulates C++ worker pool calling compile_now concurrently. With the
        # GIL each call is serialized but compile_count should still be 2.
        backend = FakeBackend(compile_delay_s=0.05)
        results = []

        def worker(k):
            results.append(backend.compile_now(k, require_reasoning=False))

        threads = [
            threading.Thread(target=worker, args=(("json", "{}"),)),
            threading.Thread(target=worker, args=(("regex", "ab*"),)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(results), 2)
        self.assertEqual(backend.compile_count, 2)


class TestSetCacheRoundTrip(unittest.TestCase):
    def test_cached_object_persists_across_calls(self):
        backend = FakeBackend()
        backend.set_cache(("ebnf", "root ::= 'x'"), FakeGrammarObject("ebnf:x"))

        first = backend.get_cached(("ebnf", "root ::= 'x'"), require_reasoning=False)
        second = backend.get_cached(("ebnf", "root ::= 'x'"), require_reasoning=False)
        self.assertEqual(first.tag, "ebnf:x")
        self.assertEqual(second.tag, "ebnf:x")

    def test_cache_overwrite(self):
        backend = FakeBackend()
        backend.set_cache(("json", "{}"), FakeGrammarObject("v1"))
        backend.set_cache(("json", "{}"), FakeGrammarObject("v2"))

        got = backend.get_cached(("json", "{}"), require_reasoning=False)
        self.assertEqual(got.tag, "v2")


if __name__ == "__main__":
    unittest.main()
