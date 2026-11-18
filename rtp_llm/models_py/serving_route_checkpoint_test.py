"""Actual CPU checkpoint I/O at the production publication boundary."""

import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.downstream_heads import ReplicatedHandlerHeads
from rtp_llm.models_py.fst_range_reader import iter_fst_range_tensors
from rtp_llm.models_py.model_loader import (
    NewLoaderConfig,
    NewLoaderLoadMethod,
    NewModelLoader,
)
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.read_spec import ReadSpec, unwrap_checkpoint_tensor
from rtp_llm.models_py.registry import register_model
from rtp_llm.models_py.weight_mapper import WeightsMapper


@register_model("_service_direct_handler")
class HandlerCheckpoint(RtpModule):
    supports_custom_weight_mappings = True
    WEIGHTS_MAPPER = WeightsMapper(prefix_mapping={"model.": ""})

    def __init__(self, config, load_config):
        super().__init__()
        self.heads = ReplicatedHandlerHeads(
            load_config.custom_weight_mappings,
            self.WEIGHTS_MAPPER,
            2,
            load_config.compute_dtype,
        )

    def load_weights(self, weights):
        for name, tensor in self.WEIGHTS_MAPPER.apply(weights):
            self.heads.capture(name, tensor)

    def validate_weights_loaded(self, loaded_tensor_ids=None):
        self.heads.validate()
        super().validate_weights_loaded(loaded_tensor_ids)


@register_model("_service_direct_leaf")
class ServiceLeaf(RtpModule):
    def __init__(self, config, load_config):
        super().__init__()
        self.weight_param = torch.nn.Parameter(torch.empty(2, 2), requires_grad=False)
        self.load_config = load_config

    def runtime_weight_view(self):
        return {"lm_head": self.weight_param}


def service(path, method, model_type="_service_direct_leaf"):
    parallel = SimpleNamespace(
        world_size=1,
        world_rank=0,
        local_rank=0,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        pp_size=1,
        pp_rank=0,
        dp_size=1,
        dp_rank=0,
        get_attn_tp_size=lambda: 1,
        get_attn_tp_rank=lambda: 0,
        get_ffn_tp_size=lambda: 1,
        get_ffn_tp_rank=lambda: 0,
    )
    return SimpleNamespace(
        model_config=SimpleNamespace(
            ckpt_path=path,
            model_type=model_type,
            num_layers=1,
            compute_dtype=torch.float32,
            quant_config=None,
            lora_infos={},
            eplb_config=SimpleNamespace(enable_eplb=lambda: False),
        ),
        _get_device_str=lambda: "cpu",
        _weight_alias_names=(),
        _weight_alias_owner=None,
        parallelism_config=parallel,
        hw_kernel_config=SimpleNamespace(use_swizzleA=False),
        moe_config=None,
        fmha_config=None,
        max_generate_batch_size=7,
        load_method=method,
        custom_module=None,
        weight=None,
        py_model=None,
        model_weights_loader=None,
        py_eplb=None,
    )


class DirectServingBoundaryTest(unittest.TestCase):
    def test_offline_conversion_keeps_owner_without_publishing_runtime(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                tensor = torch.tensor([[3.0, 4.0], [-2.0, 1.0]])
                save_file(
                    {"weight_param": tensor}, os.path.join(root, "model.safetensors")
                )
                model = service(root, method)
                load_owned_model(model, publish_runtime=False)
                self.assertIsNone(model.py_model)
                self.assertIsNone(model.weight)
                self.assertIsNone(model.py_eplb)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                torch.testing.assert_close(model.checkpoint_owner.weight_param, tensor)

    def test_forced_cpu_stages_with_native_fst_and_never_retries(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            source = torch.tensor([[2.0, 4.0], [-1.0, 3.0]])
            save_file({"weight_param": source}, path)
            value = service(root, "fastsafetensors")
            value.force_cpu_load_weights = True
            with patch.object(
                NewModelLoader,
                "_fastsafetensors_weights_iterator",
                side_effect=AssertionError("parallel FST used"),
            ), patch(
                "rtp_llm.models_py.weight_mapper.get_all_weights",
                side_effect=AssertionError("scratch used"),
            ):
                load_owned_model(value)
                self.assertEqual(
                    value.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod.FASTSAFETENSORS,
                )
                torch.testing.assert_close(value.py_model.weight_param, source)
                failed = service(root, "fastsafetensors")
                failed.force_cpu_load_weights = True
                with patch(
                    "rtp_llm.models_py.fst_range_reader.native_cpu_range_api",
                    side_effect=ImportError("reader missing"),
                ):
                    with self.assertRaisesRegex(RuntimeError, "reader missing"):
                        load_owned_model(failed)
                self.assertIsNone(failed.py_model)

    def test_native_cpu_staging_reads_pruned_vocab_rows_including_empty_rank(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "model.safetensors")
            source = torch.arange(36, dtype=torch.float32).reshape(9, 4)
            save_file({"head": source}, path)
            for rows in ((0, 1, 4, 6, 7), ()):
                with self.subTest(rows=rows):
                    values = list(
                        iter_fst_range_tensors(
                            [path],
                            read_spec_resolver=lambda key, shape: ReadSpec(
                                key, key, shape, row_indices=rows
                            ),
                        )
                    )
                    self.assertEqual(len(values), 1)
                    tensor, spec = unwrap_checkpoint_tensor(values[0][1])
                    self.assertEqual(spec.row_indices, rows)
                    torch.testing.assert_close(tensor, source[list(rows)])

    def test_full_handler_sources_survive_reader_destruction(self):
        values = {
            "model.embed_tokens.weight": torch.arange(8).reshape(4, 2).float(),
            "classifier.weight": torch.arange(6).reshape(3, 2).float(),
            "classifier.bias": torch.tensor([3.0, 1.0, -1.0]),
        }
        mappings = tuple(("__custom__." + key, key) for key in values)
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                cfg = service(root, method, "_service_direct_handler").model_config
                path = os.path.join(root, "model.safetensors")
                save_file(values, path)
                load = NewLoaderConfig(
                    device="cpu",
                    load_method=method,
                    compute_dtype=torch.float32,
                    custom_weight_mappings=mappings,
                )
                loader = NewModelLoader(cfg, load, model_path=root)
                model = loader.load()
                os.remove(path)
                self.assertEqual(loader.loaded_method, NewLoaderLoadMethod(method))
                for name, value in values.items():
                    torch.testing.assert_close(
                        model.heads.exports()["__custom__." + name], value
                    )
                bad = dict(values)
                del bad["classifier.bias"]
                save_file(bad, path)
                with self.assertRaisesRegex(RuntimeError, "Missing downstream"):
                    NewModelLoader(cfg, load, model_path=root).load()

    def test_exact_backend_and_publication_preserve_identity(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                values = torch.tensor([[2.0, 5.0], [-3.0, 7.0]])
                file = os.path.join(root, "model.safetensors")
                save_file({"weight_param": values}, file)
                model = service(root, method)
                load_owned_model(model)
                os.remove(file)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                self.assertIs(
                    model.weight.get_global_weight("lm_head"),
                    model.py_model.weight_param,
                )
                self.assertIs(model.py_model.weight, model.weight)
                self.assertIs(
                    model.py_model.load_config.py_hw_kernel_config,
                    model.hw_kernel_config,
                )
                self.assertEqual(model.py_model.load_config.max_generate_batch_size, 7)
                self.assertTrue(torch.equal(model.py_model.weight_param, values))

    def test_explicit_fst_never_retries_scratch(self):
        with tempfile.TemporaryDirectory() as root:
            torch.save(
                {"weight_param": torch.ones(2, 2)},
                os.path.join(root, "pytorch_model.bin"),
            )
            model = service(root, "fastsafetensors")
            with self.assertRaisesRegex((ValueError, RuntimeError), "safetensors"):
                load_owned_model(model)
            self.assertIsNone(model.py_model)
            self.assertIsNone(model.weight)
            self.assertIsNone(model.model_weights_loader)

    def test_failed_handler_is_not_published(self):
        class FailingHandler:
            def get_custom_weight_info(self):
                return ()

            def init(self, weights):
                raise ValueError("invalid handler weights")

        with tempfile.TemporaryDirectory() as root:
            save_file(
                {"weight_param": torch.ones(2, 2)},
                os.path.join(root, "model.safetensors"),
            )
            model = service(root, "scratch")
            model.custom_module = FailingHandler()
            with self.assertRaisesRegex(ValueError, "invalid handler weights"):
                load_owned_model(model)
            self.assertIsNone(model.weight)
            self.assertIsNone(model.py_model)


if __name__ == "__main__":
    unittest.main()
