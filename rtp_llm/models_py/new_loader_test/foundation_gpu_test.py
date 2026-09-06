import os
import tempfile
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn
from rtp_llm.models_py.layers.linear import ColumnParallelLinear
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.quant_methods.base import QuantizationConfig
from rtp_llm.models_py.registry import register_model
from safetensors.torch import save_file


class _GpuModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(2))
        self.register_buffer("scale", torch.empty(1), persistent=True)
        self.validation_devices = []

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.validation_devices.append(self.scale.device.type)
        super().validate_weights_loaded(loaded_tensor_ids)


register_model("foundation_gpu_model")(_GpuModel)


class _GpuAliasModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        same = torch.empty(1)
        self.register_buffer("same", same, persistent=True)
        self.register_buffer("same_alias", same, persistent=True)
        self.child = RtpModule()
        cross = torch.empty(1)
        self.register_buffer("cross", cross, persistent=True)
        self.child.register_buffer("cross_alias", cross, persistent=True)
        mixed = nn.Parameter(torch.empty(1))
        self.mixed_parameter = mixed
        self.register_buffer("mixed_buffer", mixed, persistent=True)
        parent_first = nn.Parameter(torch.empty(1))
        self.register_buffer("parent_buffer", parent_first, persistent=True)
        self.child.parent_parameter = parent_first
        self.left = RtpModule()
        self.right = RtpModule()
        sibling = nn.Parameter(torch.empty(1))
        self.left.register_buffer("sibling_buffer", sibling, persistent=True)
        self.right.sibling_parameter = sibling


register_model("foundation_gpu_alias_model")(_GpuAliasModel)


class _GpuStagedLeaf(RtpModule):
    def __init__(self, index, events):
        super().__init__()
        self.index = index
        self.events = events
        self.weight = nn.Parameter(
            torch.empty(1024, 1024, dtype=torch.bfloat16), requires_grad=False
        )
        self._post_load_done = False

    def requires_staged_device_postprocess(self):
        return True

    def checkpoint_load_complete(self):
        return "weight" in getattr(self, "_rtp_loaded_weight_names", set())

    def process_weights_after_loading(self):
        if self._post_load_done:
            return
        self.events.append((self.index, self.weight.device.type))
        compressed = self.weight.detach().to(torch.uint8)
        del self.weight
        self.register_parameter("weight", nn.Parameter(compressed, requires_grad=False))
        self._post_load_done = True


class _GpuFSTStagedModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.events = []
        self.layers = nn.ModuleList(
            [_GpuStagedLeaf(index, self.events) for index in range(2)]
        )


register_model("foundation_gpu_fst_staged_model")(_GpuFSTStagedModel)


class _GpuFSTOnlineQuantModel(RtpModule):
    def __init__(self, model_config, load_config):
        super().__init__()
        self.linear = ColumnParallelLinear(
            input_size=1024,
            output_size=1024,
            quant_config=QuantizationConfig("fp8_online"),
            prefix="linear",
            params_dtype=torch.bfloat16,
        )


register_model("foundation_gpu_fst_online_quant_model")(_GpuFSTOnlineQuantModel)


class FoundationGpuTest(unittest.TestCase):
    def test_fastsafetensors_preparation_preserves_buffers_and_aliases(self):
        class PreparedModel(RtpModule):
            def __init__(self):
                super().__init__()
                self.left = RtpModule()
                self.right = RtpModule()
                shared = nn.Parameter(torch.full((1024,), 11.0))
                self.left.weight = shared
                self.right.weight = shared
                self.register_buffer(
                    "derived_buffer",
                    torch.arange(8, dtype=torch.float32),
                    persistent=False,
                )

        model = PreparedModel()
        expected_buffer = model.derived_buffer.clone()
        NewModelLoader._prepare_model_for_fastsafetensors(model, "cuda:0")

        self.assertEqual(model.left.weight.device.type, "cuda")
        self.assertEqual(model.derived_buffer.device.type, "cuda")
        self.assertTrue(torch.equal(model.derived_buffer.cpu(), expected_buffer))
        self.assertIs(model.left.weight, model.right.weight)
        self.assertIsInstance(model.left.weight, nn.Parameter)

    def test_persistent_buffer_marker_survives_device_migration(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"weight": torch.ones(2), "scale": torch.ones(1)},
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                types.SimpleNamespace(model_type="foundation_gpu_model"),
                NewLoaderConfig(
                    device="cuda:0",
                    compute_dtype=torch.float32,
                    load_method=NewLoaderLoadMethod.SCRATCH,
                ),
                model_path=model_path,
            ).load()
        self.assertEqual(model.scale.device.type, "cuda")
        self.assertEqual(model.validation_devices, ["cpu", "cuda"])

    def test_buffer_aliases_survive_device_migration(self):
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {
                    "same_alias": torch.ones(1),
                    "child.cross_alias": torch.ones(1),
                    "mixed_buffer": torch.ones(1),
                    "parent_buffer": torch.ones(1),
                    "left.sibling_buffer": torch.ones(1),
                },
                os.path.join(model_path, "model.safetensors"),
            )
            model = NewModelLoader(
                types.SimpleNamespace(model_type="foundation_gpu_alias_model"),
                NewLoaderConfig(
                    device="cuda:0", load_method=NewLoaderLoadMethod.SCRATCH
                ),
                model_path=model_path,
            ).load()
        self.assertIs(model.same, model.same_alias)
        self.assertIs(model.cross, model.child.cross_alias)
        self.assertIs(model.mixed_parameter, model.mixed_buffer)
        self.assertIs(model.parent_buffer, model.child.parent_parameter)
        self.assertIs(model.left.sibling_buffer, model.right.sibling_parameter)
        self.assertIsInstance(model.child.parent_parameter, nn.Parameter)
        self.assertIsInstance(model.right.sibling_parameter, nn.Parameter)
        with torch.inference_mode():
            model.same.add_(1)
            model.child.cross_alias.add_(2)
            model.mixed_parameter.add_(3)
        self.assertEqual(model.same_alias.item(), 2)
        self.assertEqual(model.cross.item(), 3)
        self.assertEqual(model.mixed_buffer.item(), 4)
        with torch.inference_mode():
            model.to(dtype=torch.float16)
        self.assertIs(model.parent_buffer, model.child.parent_parameter)
        self.assertIs(model.left.sibling_buffer, model.right.sibling_parameter)
        self.assertIsInstance(model.child.parent_parameter, nn.Parameter)
        self.assertIsInstance(model.right.sibling_parameter, nn.Parameter)

    def test_cross_device_copy_does_not_allocate_full_target_temporary(self):
        class CudaTarget(RtpModule):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(16 * 1024 * 1024, device="cuda", dtype=torch.float32)
                )

        model = CudaTarget()
        source = torch.ones(model.weight.shape, dtype=torch.float32)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        model.load_weights({"weight": source})
        torch.cuda.synchronize()
        extra_peak = torch.cuda.max_memory_allocated() - baseline
        self.assertLess(extra_peak, 8 * 1024 * 1024)

    def test_shared_weight_migration_allocates_one_gpu_copy(self):
        class LargeTiedModel(RtpModule):
            def __init__(self):
                super().__init__()
                self.embed = RtpModule()
                self.head = RtpModule()
                shared = nn.Parameter(torch.empty(16 * 1024 * 1024))
                self.embed.weight = shared
                self.head.weight = shared
                self.register_buffer("weight_alias", shared, persistent=True)

        model = LargeTiedModel()
        weight_bytes = model.embed.weight.numel() * model.embed.weight.element_size()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()

        with torch.inference_mode():
            model.to("cuda")

        torch.cuda.synchronize()
        extra_peak = torch.cuda.max_memory_allocated() - baseline
        self.assertLess(extra_peak, weight_bytes * 2)
        self.assertIs(model.embed.weight, model.head.weight)
        self.assertIs(model.embed.weight, model.weight_alias)
        self.assertIsInstance(model.embed.weight, nn.Parameter)

    def test_staged_postprocess_bounds_online_quantization_peak_memory(self):
        class StagedLinear(RtpModule):
            def __init__(self, numel):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(numel, dtype=torch.bfloat16), requires_grad=False
                )

            def requires_staged_device_postprocess(self):
                return True

            def process_weights_after_loading(self):
                compressed = self.weight.detach().to(torch.uint8)
                del self.weight
                self.register_parameter(
                    "weight", nn.Parameter(compressed, requires_grad=False)
                )

        layer_count = 4
        layer_numel = 16 * 1024 * 1024
        model = RtpModule()
        model.layers = nn.ModuleList(
            [StagedLinear(layer_numel) for _ in range(layer_count)]
        )
        full_bf16_bytes = (
            layer_count
            * layer_numel
            * torch.empty((), dtype=torch.bfloat16).element_size()
        )

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        with torch.inference_mode():
            NewModelLoader._migrate_staged_modules(model, "cuda")
            model.to("cuda")
        torch.cuda.synchronize()
        extra_peak = torch.cuda.max_memory_allocated() - baseline

        self.assertLess(extra_peak, full_bf16_bytes)
        self.assertTrue(
            all(layer.weight.dtype == torch.uint8 for layer in model.layers)
        )

    def test_real_fastsafetensors_stream_postprocesses_staged_leaves(self):
        with tempfile.TemporaryDirectory() as model_path:
            checkpoint_weights = {
                "layers.0.weight": torch.full((1024, 1024), 1, dtype=torch.bfloat16),
                "layers.1.weight": torch.full((1024, 1024), 2, dtype=torch.bfloat16),
            }
            save_file(
                checkpoint_weights,
                os.path.join(model_path, "model.safetensors"),
            )
            with mock.patch.dict(os.environ, {"FASTSAFETENSORS_NOGDS": "1"}):
                model = NewModelLoader(
                    types.SimpleNamespace(model_type="foundation_gpu_fst_staged_model"),
                    NewLoaderConfig(
                        device="cuda:0",
                        compute_dtype=torch.bfloat16,
                        load_method=NewLoaderLoadMethod.FASTSAFETENSORS,
                    ),
                    model_path=model_path,
                ).load()

        self.assertEqual(model.events, [(0, "cuda"), (1, "cuda")])
        for index, layer in enumerate(model.layers, start=1):
            self.assertEqual(layer.weight.device.type, "cuda")
            self.assertEqual(layer.weight.dtype, torch.uint8)
            self.assertTrue(torch.all(layer.weight == index))
            self.assertFalse(hasattr(layer, "_rtp_staged_load_coordinator"))

    def test_real_fastsafetensors_runs_online_fp8_quantization(self):
        source_weight = (
            torch.linspace(
                -3.0,
                3.0,
                1024 * 1024,
                dtype=torch.float32,
            )
            .reshape(1024, 1024)
            .to(torch.bfloat16)
        )
        with tempfile.TemporaryDirectory() as model_path:
            save_file(
                {"linear.weight": source_weight},
                os.path.join(model_path, "model.safetensors"),
            )
            with mock.patch.dict(os.environ, {"FASTSAFETENSORS_NOGDS": "1"}):
                model = NewModelLoader(
                    types.SimpleNamespace(
                        model_type="foundation_gpu_fst_online_quant_model"
                    ),
                    NewLoaderConfig(
                        device="cuda:0",
                        compute_dtype=torch.bfloat16,
                        load_method=NewLoaderLoadMethod.FASTSAFETENSORS,
                    ),
                    model_path=model_path,
                ).load()

        linear = model.linear
        self.assertEqual(linear.weight.device.type, "cuda")
        self.assertEqual(linear.weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(linear.weight_scale.device.type, "cuda")
        self.assertEqual(linear.weight_scale.dtype, torch.float32)
        self.assertTrue(linear._post_load_done)
        restored = linear.weight.float().cpu() * linear.weight_scale.item()
        torch.testing.assert_close(
            restored,
            source_weight.float(),
            rtol=0.07,
            atol=0.01,
        )
        self.assertFalse(hasattr(linear, "_rtp_staged_load_coordinator"))


if __name__ == "__main__":
    unittest.main()
