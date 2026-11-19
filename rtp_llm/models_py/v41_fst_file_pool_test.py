import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.model_loader import NewLoaderConfig, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.v41_fst_file_pool import v41_file_staging_loader


class _Owner(RtpModule):
    def __init__(self):
        super().__init__()
        for name in ("first", "second"):
            self.register_parameter(name, torch.nn.Parameter(torch.empty(8), requires_grad=False))

    def checkpoint_weight_name_filter(self):
        return {"first", "second"}.__contains__


class V41FilePoolTest(unittest.TestCase):
    def test_no_pool_outside_v41_native_cuda_file_reader(self):
        class Reader:
            def _load_single_batch(self):
                pass

        for name, dev, backend, shm in (
            ("qwen3", "cuda:0", "Cuda", False),
            ("deepseek_v41", "cuda:0", "ROCm", False),
            ("deepseek_v41", "cuda:0", "Ppu", False),
            ("deepseek_v41", "cpu", "Cuda", False),
            ("deepseek_v41", "cuda:0", "Cuda", True),
        ):
            with self.subTest(name=name, dev=dev, backend=backend, shm=shm):
                self.assertIs(v41_file_staging_loader(Reader, name, torch.device(dev), cuda_backend=backend, use_shm=shm), Reader)
        for getter in (None, lambda: "cudaMallocAsync"):
            with patch.object(torch.cuda, "get_allocator_backend", getter, create=True):
                self.assertIs(v41_file_staging_loader(Reader, "deepseek_v41", torch.device("cuda:0"), cuda_backend="Cuda", use_shm=False), Reader)
        with patch.object(torch.cuda, "get_allocator_backend", side_effect=RuntimeError("unavailable"), create=True):
            self.assertIs(v41_file_staging_loader(Reader, "deepseek_v41", torch.device("cuda:0"), cuda_backend="Cuda", use_shm=False), Reader)

    def test_real_native_fst_releases_raw_files_without_flushing_model_pool(self):
        self.assertTrue(torch.cuda.is_available())
        self.assertEqual(torch.cuda.get_allocator_backend(), "native")
        # Keep a pre-existing allocator segment alive but cached. It must not
        # be purged just to satisfy a subsequent cold-start memory budget.
        live_param = torch.arange(8., device="cuda")
        old_cache = torch.empty(4 << 20, device="cuda", dtype=torch.uint8)
        old_segment = old_cache.data_ptr()
        del old_cache
        expected = torch.arange(8.)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, name in enumerate(("first", "second")):
                save_file(
                    {name: expected + i, f"unused{i}": torch.zeros(64 << 20, dtype=torch.uint8)},
                    str(root / f"{i}.safetensors"),
                )
            def read(model_type):
                loader = NewModelLoader(
                    {"model_type": model_type},
                    NewLoaderConfig(device="cuda:0", load_method="fastsafetensors"),
                    model_path=tmp,
                )
                with patch.object(loader, "_create_model", side_effect=_Owner):
                    model = loader.load()
                self.assertEqual(loader.loaded_method.value, "fastsafetensors")
                self.assertTrue(model.first.is_cuda and model.second.is_cuda)
                torch.testing.assert_close(model.first.cpu(), expected)
                torch.testing.assert_close(model.second.cpu(), expected + 1)
                return model

            def reserved():
                torch.cuda.synchronize()
                return torch.cuda.memory_reserved()

            initial = reserved()
            pooled = read("deepseek_v41")
            after = reserved()
            unpooled = read("ordinary")
            control = reserved()
            print(f"FST raw file reserved bytes: initial={initial} V4.1={after} control={control}")
            self.assertLess(after - initial, 64 << 20)
            self.assertGreaterEqual(control - after, 64 << 20)
            self.assertTrue(any(
                seg["address"] <= old_segment < seg["address"] + seg["total_size"]
                for seg in torch.cuda.memory_snapshot() if seg["device"] == 0
            ))
            torch.testing.assert_close(live_param.cpu(), expected)
            torch.testing.assert_close(pooled.first, unpooled.first)


if __name__ == "__main__":
    unittest.main()
