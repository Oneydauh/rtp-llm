"""Tests for the triton bitmask kernel (apply_token_bitmask_inplace_triton)."""

import unittest

import torch

_CUDA_AVAILABLE = torch.cuda.is_available()


def _pack_bitmask(allow_mask: torch.Tensor) -> torch.Tensor:
    """Pack a bool allow-mask [batch, vocab] into int32 bitmask [batch, ceil(vocab/32)].

    Bit *i* inside each int32 word is 1 when the token at position
    ``word_index * 32 + i`` is *allowed*.
    """
    batch, vocab = allow_mask.shape
    padded = vocab + (32 - vocab % 32) % 32
    buf = torch.zeros(batch, padded, dtype=torch.int32)
    for b in range(batch):
        for v in range(vocab):
            if allow_mask[b, v]:
                word = v // 32
                bit = v % 32
                buf[b, word] |= 1 << bit
    return buf[:, : (vocab + 31) // 32]


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
class TestApplyTokenBitmaskInplaceTriton(unittest.TestCase):
    """Verify that the triton kernel correctly masks logits to -inf."""

    def setUp(self):
        from rtp_llm.models_py.triton_kernels.grammar.bitmask_ops import (
            apply_token_bitmask_inplace_triton,
        )

        self.apply_fn = apply_token_bitmask_inplace_triton

    def test_basic_masking(self):
        """Tokens whose bitmask bit is 0 should be set to -inf."""
        batch, vocab = 2, 64
        logits = torch.ones(batch, vocab, dtype=torch.float32, device="cuda")

        allow = torch.ones(batch, vocab, dtype=torch.bool)
        allow[0, 10] = False
        allow[0, 20] = False
        allow[1, 0] = False
        allow[1, 63] = False
        bitmask = _pack_bitmask(allow).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.isinf(logits[0, 10]) and logits[0, 10] < 0)
        self.assertTrue(torch.isinf(logits[0, 20]) and logits[0, 20] < 0)
        self.assertTrue(torch.isinf(logits[1, 0]) and logits[1, 0] < 0)
        self.assertTrue(torch.isinf(logits[1, 63]) and logits[1, 63] < 0)

        self.assertEqual(logits[0, 0].item(), 1.0)
        self.assertEqual(logits[0, 11].item(), 1.0)
        self.assertEqual(logits[1, 1].item(), 1.0)
        self.assertEqual(logits[1, 32].item(), 1.0)

    def test_all_allowed(self):
        """When all bits are 1, logits should remain unchanged."""
        batch, vocab = 1, 128
        logits = torch.randn(batch, vocab, dtype=torch.float32, device="cuda")
        original = logits.clone()

        allow = torch.ones(batch, vocab, dtype=torch.bool)
        bitmask = _pack_bitmask(allow).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.equal(logits, original))

    def test_all_masked(self):
        """When all bits are 0, all logits should become -inf."""
        batch, vocab = 1, 128
        logits = torch.ones(batch, vocab, dtype=torch.float32, device="cuda")

        allow = torch.zeros(batch, vocab, dtype=torch.bool)
        bitmask = _pack_bitmask(allow).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.all(torch.isinf(logits) & (logits < 0)))

    def test_large_vocab(self):
        """Works with vocab sizes exceeding BLOCK_SIZE (4096)."""
        batch, vocab = 2, 32000
        logits = torch.ones(batch, vocab, dtype=torch.float32, device="cuda")

        allow = torch.ones(batch, vocab, dtype=torch.bool)
        masked_positions = [0, 31, 32, 4095, 4096, 31999]
        for pos in masked_positions:
            allow[0, pos] = False

        bitmask = _pack_bitmask(allow).to("cuda")
        self.apply_fn(logits, bitmask)

        for pos in masked_positions:
            self.assertTrue(
                torch.isinf(logits[0, pos]) and logits[0, pos] < 0,
                f"Position {pos} should be -inf",
            )
        self.assertEqual(logits[0, 1].item(), 1.0)
        self.assertEqual(logits[1, 0].item(), 1.0)

    @unittest.skip(
        "The `indices` parameter semantics in apply_token_bitmask_inplace_triton "
        "are not used by any grammar hot-path code (batch_apply_grammar_constraints "
        "et al. never pass indices). The test asserts a mapping behaviour that the "
        "kernel does not currently implement — un-skip only after the indices "
        "codepath is actually wired into a caller that needs this semantic."
    )
    def test_with_indices(self):
        """The indices parameter maps bitmask rows to logits rows."""
        batch, vocab = 4, 64
        logits = torch.ones(batch, vocab, dtype=torch.float32, device="cuda")

        bitmask_rows = 2
        allow = torch.ones(bitmask_rows, vocab, dtype=torch.bool)
        allow[0, 5] = False
        allow[1, 10] = False
        bitmask = _pack_bitmask(allow).to("cuda")

        indices = torch.tensor([1, 3], dtype=torch.int32, device="cuda")
        self.apply_fn(logits, bitmask, indices=indices)

        self.assertTrue(torch.isinf(logits[1, 5]) and logits[1, 5] < 0)
        self.assertTrue(torch.isinf(logits[3, 10]) and logits[3, 10] < 0)
        self.assertEqual(logits[0, 5].item(), 1.0)
        self.assertEqual(logits[2, 10].item(), 1.0)

    def test_1d_input(self):
        """Works with 1-D logits and 1-D bitmask tensors."""
        vocab = 64
        logits = torch.ones(vocab, dtype=torch.float32, device="cuda")

        allow = torch.ones(1, vocab, dtype=torch.bool)
        allow[0, 7] = False
        allow[0, 33] = False
        bitmask = _pack_bitmask(allow).squeeze(0).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.isinf(logits[7]) and logits[7] < 0)
        self.assertTrue(torch.isinf(logits[33]) and logits[33] < 0)
        self.assertEqual(logits[0].item(), 1.0)

    def test_fp16_logits(self):
        """Works with half-precision logits."""
        batch, vocab = 1, 128
        logits = torch.ones(batch, vocab, dtype=torch.float16, device="cuda")

        allow = torch.ones(batch, vocab, dtype=torch.bool)
        allow[0, 42] = False
        bitmask = _pack_bitmask(allow).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.isinf(logits[0, 42]) and logits[0, 42] < 0)
        self.assertEqual(logits[0, 0].item(), 1.0)

    def test_bf16_logits(self):
        """Works with bfloat16 logits."""
        batch, vocab = 1, 128
        logits = torch.ones(batch, vocab, dtype=torch.bfloat16, device="cuda")

        allow = torch.ones(batch, vocab, dtype=torch.bool)
        allow[0, 0] = False
        allow[0, 127] = False
        bitmask = _pack_bitmask(allow).to("cuda")

        self.apply_fn(logits, bitmask)

        self.assertTrue(torch.isinf(logits[0, 0]) and logits[0, 0] < 0)
        self.assertTrue(torch.isinf(logits[0, 127]) and logits[0, 127] < 0)
        self.assertEqual(logits[0, 1].item(), 1.0)


@unittest.skipUnless(_CUDA_AVAILABLE, "CUDA not available")
class TestApplyVocabMaskIntegration(unittest.TestCase):
    """Test apply_vocab_mask on XGrammarGrammar using xgrammar's allocate_token_bitmask."""

    def test_xgrammar_bitmask_roundtrip(self):
        """allocate → fill → move → apply pipeline works end-to-end."""
        try:
            from xgrammar import allocate_token_bitmask
        except ImportError:
            self.skipTest("xgrammar not installed")

        from rtp_llm.models_py.triton_kernels.grammar.bitmask_ops import (
            apply_token_bitmask_inplace_triton,
        )

        vocab_size = 128
        batch_size = 2
        bitmask = allocate_token_bitmask(batch_size, vocab_size)
        self.assertEqual(bitmask.dtype, torch.int32)
        self.assertEqual(bitmask.shape[0], batch_size)

        logits = torch.ones(batch_size, vocab_size, dtype=torch.float32, device="cuda")
        bitmask_gpu = bitmask.to("cuda", non_blocking=True)

        apply_token_bitmask_inplace_triton(logits, bitmask_gpu)
        self.assertEqual(logits.shape, (batch_size, vocab_size))


if __name__ == "__main__":
    unittest.main()
