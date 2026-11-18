"""Actual CPU checkpoint I/O at the production publication boundary."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.direct_checkpoint import load_owned_model
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


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
