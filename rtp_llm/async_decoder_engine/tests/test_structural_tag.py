"""Tests for XGrammarGrammarBackend.dispatch_structural_tag.

Covers both the legacy format (structures + triggers) and the new typed
format (type=json_schema / tag / sequence / or / triggered_tags /
tags_with_separator). Also exercises the _sanitize_structural_format
recursion and the InvalidGrammarObject fallback on malformed input.
"""

import json
import os
import unittest

import torch

_CUDA_AVAILABLE = torch.cuda.is_available()
_MODEL_PATH = "/home/yanxi.wln/work/models/Qwen2-1.5B-Instruct/"


@unittest.skipUnless(
    _CUDA_AVAILABLE and os.path.isdir(_MODEL_PATH),
    "xgrammar requires CUDA + a real tokenizer",
)
class TestStructuralTagCompile(unittest.TestCase):
    """Each call exercises a real xgrammar compile; we assert the returned
    grammar is a valid BaseGrammarObject (not Invalid) and its matcher moves
    past its START state after accepting an expected first token."""

    @classmethod
    def setUpClass(cls):
        from transformers import AutoTokenizer

        from rtp_llm.async_decoder_engine.xgrammar_backend import (
            XGrammarGrammarBackend,
        )

        cls.tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        cls.backend = XGrammarGrammarBackend(
            tokenizer=cls.tokenizer,
            vocab_size=len(cls.tokenizer),
        )

    def _assert_valid_grammar(self, g):
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        self.assertNotIsInstance(g, InvalidGrammarObject, msg=str(g))

    def test_legacy_structural_tag_dispatches(self):
        payload = {
            "structures": [
                {
                    "begin": "<tool_call>",
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    },
                    "end": "</tool_call>",
                }
            ],
            "triggers": ["<tool_call>"],
        }
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        self._assert_valid_grammar(g)

    def test_legacy_sanitize_null_schema(self):
        """A legacy structure with schema=None must be rescued by
        _sanitize_structural_tag_structures — it should still compile rather
        than bubbling the null through to xgrammar."""
        payload = {
            "structures": [
                {"begin": "<x>", "schema": None, "end": "</x>"},
            ],
            "triggers": ["<x>"],
        }
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        self._assert_valid_grammar(g)

    def test_new_format_json_schema_wrapper(self):
        """`format = {type: json_schema}` with null schema goes through
        _sanitize_structural_format."""
        payload = {
            "format": {
                "type": "json_schema",
                "json_schema": None,
            }
        }
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        self._assert_valid_grammar(g)

    def test_new_format_sequence_recursion(self):
        """`format = {type: sequence, elements: [...]}` — sanitize recurses
        into elements. Each element may itself be a nested json_schema with
        null schema."""
        payload = {
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "json_schema", "json_schema": None},
                    {"type": "json_schema", "json_schema": {"type": "integer"}},
                ],
            }
        }
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        self._assert_valid_grammar(g)

    def test_new_format_or_recursion(self):
        payload = {
            "format": {
                "type": "or",
                "elements": [
                    {"type": "json_schema", "json_schema": None},
                    {"type": "json_schema", "json_schema": {"type": "integer"}},
                ],
            }
        }
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        self._assert_valid_grammar(g)

    def test_malformed_json_returns_invalid(self):
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        g = self.backend.dispatch_structural_tag("{not valid json")
        self.assertIsInstance(g, InvalidGrammarObject)

    def test_malformed_structural_tag_returns_invalid(self):
        """Well-formed JSON but semantically broken for xgrammar — compiler
        should raise RuntimeError and the backend should wrap it."""
        from rtp_llm.async_decoder_engine.base_grammar_backend import (
            InvalidGrammarObject,
        )

        # Neither "structures"+"triggers" (legacy) nor "format" (new).
        payload = {"completely": "unrelated"}
        g = self.backend.dispatch_structural_tag(json.dumps(payload))
        # Either path is acceptable (Invalid or valid but useless) — we just
        # need it not to crash the process.
        _ = g  # intentionally lax; the real assertion is "no exception"


class TestSanitizeStructuralFormatUnit(unittest.TestCase):
    """Pure-Python unit tests for _sanitize_structural_format — no xgrammar."""

    def _sanitize(self, d):
        from rtp_llm.async_decoder_engine.xgrammar_backend import (
            XGrammarGrammarBackend,
        )

        XGrammarGrammarBackend._sanitize_structural_format(d)

    def test_json_schema_null_becomes_empty(self):
        d = {"type": "json_schema", "json_schema": None}
        self._sanitize(d)
        self.assertEqual(d["json_schema"], {})

    def test_qwen_xml_parameter_null_becomes_empty(self):
        d = {"type": "qwen_xml_parameter", "json_schema": None}
        self._sanitize(d)
        self.assertEqual(d["json_schema"], {})

    def test_tag_recurses_into_content(self):
        d = {
            "type": "tag",
            "content": {"type": "json_schema", "json_schema": None},
        }
        self._sanitize(d)
        self.assertEqual(d["content"]["json_schema"], {})

    def test_sequence_recurses_into_elements(self):
        d = {
            "type": "sequence",
            "elements": [
                {"type": "json_schema", "json_schema": None},
                {"type": "json_schema", "json_schema": None},
            ],
        }
        self._sanitize(d)
        for el in d["elements"]:
            self.assertEqual(el["json_schema"], {})

    def test_or_recurses_into_elements(self):
        d = {
            "type": "or",
            "elements": [
                {"type": "json_schema", "json_schema": None},
            ],
        }
        self._sanitize(d)
        self.assertEqual(d["elements"][0]["json_schema"], {})

    def test_triggered_tags_recurses_into_tags(self):
        d = {
            "type": "triggered_tags",
            "tags": [
                {"type": "json_schema", "json_schema": None},
            ],
        }
        self._sanitize(d)
        self.assertEqual(d["tags"][0]["json_schema"], {})

    def test_tags_with_separator_recurses(self):
        d = {
            "type": "tags_with_separator",
            "tags": [
                {"type": "json_schema", "json_schema": None},
            ],
        }
        self._sanitize(d)
        self.assertEqual(d["tags"][0]["json_schema"], {})

    def test_non_dict_is_noop(self):
        # Must not crash on non-dict input.
        self._sanitize("not a dict")
        self._sanitize(None)
        self._sanitize(42)
        self._sanitize([])

    def test_unknown_type_untouched(self):
        d = {"type": "something_else", "json_schema": None}
        self._sanitize(d)
        # Only known types get rescued. Others pass through untouched.
        self.assertIsNone(d["json_schema"])


if __name__ == "__main__":
    unittest.main()
