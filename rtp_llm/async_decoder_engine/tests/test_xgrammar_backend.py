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

        mock_tokenizer = MagicMock()
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


if __name__ == "__main__":
    unittest.main()
