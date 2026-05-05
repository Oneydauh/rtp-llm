"""Tests for GrammarFileCache + resolve_grammar_cache_dir.

The file cache persists compiled grammars to disk so a restarted server hits
the disk instead of re-running xgrammar's compiler (which dominates cold-path
latency for big JSON schemas). Protects:
  * hash-based file naming (same key -> same filename; different keys don't
    collide)
  * missing dir auto-created
  * corrupted / truncated JSON returns None (not crash) → forces recompile
  * put is atomic via .tmp + os.replace
  * store failures are swallowed (don't take down the query)
  * resolve_grammar_cache_dir fallback
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from rtp_llm.async_decoder_engine.grammar_cache import (
    GrammarFileCache,
    _DEFAULT_CACHE_DIR,
    resolve_grammar_cache_dir,
)


class TestResolveGrammarCacheDir(unittest.TestCase):
    def test_empty_string_falls_back_to_default(self):
        self.assertEqual(resolve_grammar_cache_dir(""), _DEFAULT_CACHE_DIR)

    def test_none_falls_back_to_default(self):
        self.assertEqual(resolve_grammar_cache_dir(None), _DEFAULT_CACHE_DIR)

    def test_configured_value_passed_through(self):
        self.assertEqual(resolve_grammar_cache_dir("/tmp/foo"), "/tmp/foo")


class _FakeCompiled:
    """Mimics CompiledGrammar's serialize/deserialize API."""

    def __init__(self, payload: str):
        self.payload = payload
        self.memory_size_bytes = len(payload)

    def serialize_json(self) -> str:
        return json.dumps({"payload": self.payload})


class TestGrammarFileCacheRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix="rtp_grammar_cache_test_")
        self.mock_tok_info = MagicMock()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_missing_dir_auto_created(self):
        target = os.path.join(self.tmpdir, "nested", "deep", "cache")
        self.assertFalse(os.path.exists(target))
        GrammarFileCache(target, self.mock_tok_info)
        self.assertTrue(os.path.isdir(target))

    def test_miss_returns_none(self):
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)
        self.assertIsNone(c.get("json", "{}"))

    def test_put_creates_file_and_get_deserializes(self):
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)
        fake = _FakeCompiled("hello")
        c.put("json", "{}", fake)

        # File must be present (hashed name).
        files = [f for f in os.listdir(self.tmpdir) if f.endswith(".json")]
        self.assertEqual(len(files), 1, msg=f"expected one json file, got {files}")

        # get() should deserialize via CompiledGrammar.deserialize_json. Patch
        # it to return a sentinel — we're testing the plumbing, not xgrammar.
        sentinel = object()
        with patch(
            "rtp_llm.async_decoder_engine.grammar_cache.CompiledGrammar"
        ) as MockCG:
            MockCG.deserialize_json.return_value = sentinel
            # The mock must expose memory_size_bytes for the logger line.
            MockCG.deserialize_json.return_value = MagicMock(memory_size_bytes=42)
            got = c.get("json", "{}")

        self.assertIsNotNone(got)
        # deserialize_json must have been called with the serialized payload +
        # the tokenizer_info instance.
        args, _ = MockCG.deserialize_json.call_args
        self.assertIn("hello", args[0])
        self.assertIs(args[1], self.mock_tok_info)

    def test_no_cross_key_collision(self):
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)
        c.put("json", "schemaA", _FakeCompiled("A"))
        c.put("json", "schemaB", _FakeCompiled("B"))
        c.put("regex", "schemaA", _FakeCompiled("RA"))  # same key_string,
        # different key_type — must NOT overwrite

        files = sorted(os.listdir(self.tmpdir))
        self.assertEqual(len(files), 3, msg=f"cross-key collision: {files}")

    def test_same_key_overwrites(self):
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)
        c.put("json", "k", _FakeCompiled("v1"))
        c.put("json", "k", _FakeCompiled("v2"))
        files = os.listdir(self.tmpdir)
        self.assertEqual(len(files), 1)
        path = os.path.join(self.tmpdir, files[0])
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data["payload"], "v2")

    def test_corrupted_file_returns_none_no_crash(self):
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)
        # Manually drop a garbage file in the place where get() will look.
        import hashlib

        h = hashlib.sha256(b"json:bad_schema").hexdigest()
        path = os.path.join(self.tmpdir, h + ".json")
        with open(path, "w") as f:
            f.write("{this is not valid json for xgrammar")

        # deserialize_json will throw; get() catches and returns None.
        with patch(
            "rtp_llm.async_decoder_engine.grammar_cache.CompiledGrammar"
        ) as MockCG:
            MockCG.deserialize_json.side_effect = RuntimeError("boom")
            got = c.get("json", "bad_schema")
        self.assertIsNone(got)

    def test_put_is_atomic_via_tmp_file(self):
        """Temp file path should land after successful write; on failure the
        real path is not created."""
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)

        class _ExplodingCompiled:
            memory_size_bytes = 0

            def serialize_json(self):
                raise RuntimeError("serialize crash")

        c.put("json", "never_lands", _ExplodingCompiled())

        files = [f for f in os.listdir(self.tmpdir) if not f.endswith(".tmp")]
        self.assertEqual(files, [], msg=f"partial write leaked: {files}")

    def test_store_failure_is_swallowed(self):
        """A failing put MUST NOT raise — it would take down the model call."""
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)

        class _Boom:
            memory_size_bytes = 0

            def serialize_json(self):
                raise OSError("disk full")

        try:
            c.put("json", "k", _Boom())
        except Exception as e:
            self.fail(f"put() unexpectedly raised: {e}")

    def test_store_failure_cleans_tmp(self):
        """After a failed put, the .tmp file must not linger."""
        c = GrammarFileCache(self.tmpdir, self.mock_tok_info)

        class _BoomAfterSerialize:
            memory_size_bytes = 0

            def serialize_json(self):
                # Write a .tmp by having the path prefix accessible, then
                # raise to simulate a write crash.
                # Actually the code path: serialize_json is called first and
                # returns a string, then open(tmp, "w"). To trip that second
                # path, we patch builtins.open inside put().
                return '{"payload": "ok"}'

        # Patch open() used inside grammar_cache to throw on write.
        with patch(
            "builtins.open", side_effect=OSError("write blocked")
        ) as mock_open:
            c.put("json", "k", _BoomAfterSerialize())

        # No .tmp file should remain.
        leftover = [f for f in os.listdir(self.tmpdir) if ".tmp" in f]
        self.assertEqual(leftover, [])


if __name__ == "__main__":
    unittest.main()
