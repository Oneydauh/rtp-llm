import unittest

import torch

from rtp_llm.models_py.kernels.cuda.fp8_quant import per_block_cast_to_fp8


class PerBlockFp8QuantTest(unittest.TestCase):
    def test_non_ue8m0_matches_legacy_multiplier_rounding(self):
        # This pair sits exactly on the FP8 midpoint only when the legacy
        # multiplier order (448 / amax) is used. Reassociating it as
        # 1 / (amax / 448) rounds the second value to 11 instead of 10.
        amax = 4.049753189086914
        value = 0.09491609036922455
        weight = torch.zeros((128, 128), dtype=torch.float32)
        weight[0, 0] = amax
        weight[0, 1] = value

        quantized, scale = per_block_cast_to_fp8(weight, use_ue8m0=False)

        self.assertEqual(float(quantized[0, 1]), 10.0)
        torch.testing.assert_close(
            scale,
            torch.tensor([[amax / 448.0]], dtype=torch.float32),
            rtol=0.0,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
