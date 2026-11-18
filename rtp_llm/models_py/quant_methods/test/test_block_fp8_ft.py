import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.ft_checkpoint import CheckpointTopology, FTCheckpoint
from rtp_llm.models_py.quant_methods.block_fp8_ft import (
    block_ft_bindings, block_ft_scale, block_ft_weight,
)
from rtp_llm.models_py.quant_methods.fp8 import Fp8BlockLinearMethod


def write_ft(directory, data):
    keys = {"rank_0_0_0." + k: v.contiguous() for k, v in data.items()}
    save_file(keys, str(Path(directory) / "ft.safetensors"))
    (Path(directory) / "model.safetensors.index.json").write_text(json.dumps({
        "is_ft_style_weight": True, "weight_map": {k: "ft.safetensors" for k in keys}}))


def linear(n, k):
    model = torch.nn.Module()
    model.proj = torch.nn.Module()
    model.proj.quant_config = SimpleNamespace(weight_block_size=[128, 128])
    Fp8BlockLinearMethod(model.proj.quant_config).create_weights(
        model.proj, input_size=k, output_size=n, params_dtype=torch.bfloat16
    )
    return model


def source_snapshot(w, s, packed):
    n, k = w.shape
    if not packed:
        # main PerBlockFp8Weight._postprocess: flat reshape, not transpose.
        return w.reshape(k, n), s.reshape(s.shape[1], -1)
    # Same persisted int32 little-endian representation as DeepGEMM's packer;
    # this CPU fixture only exercises decoding, not the SM100 CUDA pack kernel.
    e = torch.log2(s).int() + 127
    e = e.repeat_interleave(128, dim=0)[:n]
    e = torch.nn.functional.pad(e, (0, (-e.shape[1]) % 4))
    byte = e.view(n, -1, 4)
    raw = byte[..., 0] | byte[..., 1] << 8 | byte[..., 2] << 16 | byte[..., 3] << 24
    return w.clone(), raw


class BlockFTTest(unittest.TestCase):
    def test_real_file_into_block_linear_owner_and_output(self):
        n, k = 257, 521   # partial N, K blocks and packed int32 padding
        g = torch.Generator().manual_seed(57)
        w = torch.randint(-11, 12, (n,k), generator=g).to(torch.float8_e4m3fn)
        s = (2.0 ** torch.randint(-3, 4, (3,5), generator=g)).float()
        input = torch.randn(3,k,generator=g)
        def forward(weight, scale):
            dequant = weight.float() * scale.repeat_interleave(128,0).repeat_interleave(128,1)[:n,:k]
            return input @ dequant.T
        ref = forward(w,s)
        for packed in (False,True):
            kernel,scale = source_snapshot(w,s,packed)
            with tempfile.TemporaryDirectory() as root:
                write_ft(root,{"layers.0.kernel":kernel, "layers.0.scale":scale})
                for backend in ("scratch","fastsafetensors"):
                    with self.subTest(packed=packed, backend=backend):
                        cp=FTCheckpoint.open(root, CheckpointTopology())
                        cp.read_backend=backend
                        m=linear(n,k)
                        ptr=m.proj.weight.data_ptr()
                        bs=block_ft_bindings(m.proj,"layers.0.kernel","layers.0.scale","proj",cp)
                        cp.load_parameters(m, bs)
                        self.assertEqual(ptr,m.proj.weight.data_ptr())
                        self.assertTrue(torch.equal(w.view(torch.uint8),m.proj.weight.view(torch.uint8)))
                        torch.testing.assert_close(m.proj.weight_scale_inv,s,rtol=0,atol=0)
                        torch.testing.assert_close(forward(m.proj.weight,m.proj.weight_scale_inv),ref,rtol=0,atol=0)
                        self.assertEqual(bool(m.proj._packed_block_fp8_ft),packed)

    def test_reject_malformed_packed_and_inconsistent_shape(self):
        t = torch.full((259,2), 127, dtype=torch.int32)
        for row in (2,258):
            v = t.clone()
            v[row,0] = 123
            with self.assertRaisesRegex(ValueError,"inconsistent"):
                block_ft_scale(v,(259,600),source_dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError,"NaN exponent"):
            block_ft_scale(t.fill_(-1),(259,600),source_dtype=torch.float8_e4m3fn)
        with self.assertRaisesRegex(ValueError,"weight shape"):
            block_ft_weight(torch.ones(7,2).to(torch.float8_e4m3fn),(7,2),False)

    def test_rocm_fnuz_canonicalization(self):
        w = torch.tensor([[1., 2., 4.],[-1.,-2.,-4.]]).to(torch.float8_e4m3fn)
        raw = w.view(torch.float8_e4m3fnuz).reshape(3,2)
        decoded = block_ft_weight(raw,(2,3),False)
        s = block_ft_scale(torch.tensor([[4.]]),(2,3),source_dtype=raw.dtype)
        torch.testing.assert_close(decoded.float()*s, raw.reshape(2,3).float()*4,rtol=0,atol=0)


if __name__ == "__main__":
    unittest.main()
