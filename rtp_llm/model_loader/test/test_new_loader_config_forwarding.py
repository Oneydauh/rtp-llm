import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from rtp_llm.models.base_model import BaseModel


class _FakeNewModelLoader:
    load_config = None

    def __init__(self, model_config, load_config, model_path, device):
        type(self).load_config = load_config

    def load(self):
        return object()


class NewLoaderConfigForwardingTest(unittest.TestCase):

    def test_independent_runtime_configs_are_forwarded(self):
        model = object.__new__(BaseModel)
        model.parallelism_config = SimpleNamespace(
            tp_size=2,
            tp_rank=1,
            ep_size=4,
            ep_rank=3,
        )
        model.model_config = SimpleNamespace(
            quant_config=None,
            compute_dtype=torch.bfloat16,
            ckpt_path="/tmp/model",
        )
        model.hw_kernel_config = SimpleNamespace()
        model.fmha_config = object()
        model.device_resource_config = object()
        model.moe_config = object()
        model.force_cpu_load_weights = False

        model._get_device_str = lambda: "cuda:1"
        model._get_quant_type = lambda: "none"
        model._init_custom_module = lambda: None
        model._build_weights_from_module = lambda module: object()
        model._load_custom_module = lambda: None
        model._init_multimodal_for_new_loader = lambda: None

        with mock.patch(
            "rtp_llm.models_py.model_loader.NewModelLoader",
            _FakeNewModelLoader,
        ):
            model._load_with_new_loader()

        load_config = _FakeNewModelLoader.load_config
        self.assertIs(load_config.parallelism_config, model.parallelism_config)
        self.assertIs(load_config.fmha_config, model.fmha_config)
        self.assertIs(load_config.device_resource_config, model.device_resource_config)
        self.assertIs(load_config.moe_config, model.moe_config)
        self.assertEqual(load_config.compute_dtype, torch.bfloat16)
        self.assertEqual(load_config.device, "cuda:1")


if __name__ == "__main__":
    unittest.main()
