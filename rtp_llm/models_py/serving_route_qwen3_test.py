"""Public BaseModel.from_config -> Qwen3 owner, on a real main native ABI."""

import os
import tempfile
import unittest
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models.qwen_v3 import QwenV3
from rtp_llm.models_py.model_loader import NewLoaderLoadMethod
from rtp_llm.models_py.new_models.qwen3.language import Qwen3ForCausalLM
from rtp_llm.models_py.new_models.qwen3.main_pp_checkpoint_test import (
    checkpoint,
    model_config,
)
from rtp_llm.models_py.serving_route_checkpoint_test import service


class CheckpointOnlyQwen3(QwenV3):
    """No tokenizer or GPU kernel needed for this weight-loading regression."""

    def load_tokenizer(self):
        pass

    def _get_device_str(self):
        return "cpu"

    def _init_custom_module(self):
        return None

    def create_model_loader(self):
        raise AssertionError("explicit direct route leaked into the legacy loader")

    def _create_python_model(self):
        raise AssertionError("explicit direct route reconstructed a legacy model")


class Qwen3ServiceRouteTest(unittest.TestCase):
    def _load(self, root, method):
        fixture = service(root, method)
        config = model_config()
        config.model_type = "qwen_3"
        config.ckpt_path = root
        config.quant_config = None
        config.lora_infos = {}
        config.generate_env_config = None
        config.compute_dtype = torch.float32
        config.eplb_config = fixture.model_config.eplb_config
        config.enable_output_vocab_pruning = False
        return CheckpointOnlyQwen3.from_config(
            model_config=config,
            kv_cache_config=SimpleNamespace(
                multi_task_prompt="", multi_task_prompt_str=""
            ),
            parallelism_config=fixture.parallelism_config,
            fmha_config=None,
            moe_config=None,
            max_generate_batch_size=7,
            hw_kernel_config=SimpleNamespace(
                enable_cuda_graph=False, use_swizzleA=False
            ),
            load_method=method,
            use_new_loader=True,
            merge_lora=False,
            vit_config=None,
        )

    def test_scratch_and_native_fst(self):
        for method in ("scratch", "fastsafetensors"):
            with self.subTest(method=method), tempfile.TemporaryDirectory() as root:
                data = checkpoint()
                save_file(data, os.path.join(root, "model.safetensors"))
                model = self._load(root, method)
                self.assertIsInstance(model.py_model, Qwen3ForCausalLM)
                self.assertEqual(
                    model.model_weights_loader.loaded_method,
                    NewLoaderLoadMethod(method),
                )
                self.assertIs(
                    model.py_model.embed_tokens.weight,
                    model.weight.get_global_weight("embedding"),
                )
                torch.testing.assert_close(
                    model.py_model.embed_tokens.weight,
                    data["model.embed_tokens.weight"],
                )
                self.assertIsNone(model.weight_manager)


if __name__ == "__main__":
    unittest.main()
