"""Real checkpoint -> generic stacked experts; sequential rank views, no GPU."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from rtp_llm.models_py.layers.moe_experts import BaseMoEExperts
from rtp_llm.models_py.model_loader import NewLoaderConfig, NewLoaderLoadMethod, NewModelLoader
from rtp_llm.models_py.module_base import RtpModule
from rtp_llm.models_py.registry import register_model


@register_model("_glm4_nonlite_main_expert_test")
class _ExpertModel(RtpModule):
    fastsafetensors_load_isolated = True

    def __init__(self, config, loader):
        super().__init__()
        self.experts = BaseMoEExperts(
            num_experts=4, top_k=2, hidden_size=4, moe_intermediate_size=4,
            tp_size=config.tp, tp_rank=config.tr, ep_size=config.ep, ep_rank=config.er,
            params_dtype=torch.bfloat16, model_config=config,
            parallelism_config=SimpleNamespace(dp_size=1, dp_rank=0, world_size=max(config.tp,config.ep), local_world_size=max(config.tp,config.ep)),
            moe_config=None, quant_config=None, layer_idx=0,
        )


class GenericExpertsCheckpointTest(unittest.TestCase):
    def exercise(self, method):
        src = {}
        for expert in range(4):
            for index, name in enumerate(("up", "gate", "down")):
                src[f"experts.{expert}.{name}_proj.weight"] = (torch.arange(16).reshape(4, 4) + expert * 48 + index * 16).bfloat16()
        with tempfile.TemporaryDirectory() as folder:
            save_file(src, str(Path(folder) / "model.safetensors"))
            for tp,tr,ep,er in ((1,0,1,0),(2,0,1,0),(2,1,1,0),(1,0,2,0),(1,0,2,1)):
                with self.subTest(tp=tp, tr=tr, ep=ep, er=er, method=method):
                    cfg = SimpleNamespace(model_type="_glm4_nonlite_main_expert_test",num_layers=1,activation_type="SiGLU",tp=tp,tr=tr,ep=ep,er=er)
                    loader=NewModelLoader(cfg,NewLoaderConfig(device="cpu",load_method=method),model_path=folder)
                    experts=loader.load().experts
                    self.assertEqual(loader._selected_load_method, NewLoaderLoadMethod(method))
                    self.assertTrue(experts.is_cpu_checkpoint_staging())
                    self.assertIsNone(experts.fused_moe)
                    for local in range(4//ep):
                        gid=er*(4//ep)+local
                        pair=[src[f"experts.{gid}.{name}_proj.weight"].chunk(tp,dim=0)[tr] for name in ("up","gate")]
                        torch.testing.assert_close(experts.w13[local],torch.cat(pair),rtol=0,atol=0)
                        torch.testing.assert_close(experts.w2[local],src[f"experts.{gid}.down_proj.weight"].chunk(tp,dim=1)[tr],rtol=0,atol=0)

    def test_scratch(self):
        self.exercise("scratch")

    def test_native_fst(self):
        import fastsafetensors.cpp as native
        self.assertTrue(native.__file__.endswith(".so"))
        self.exercise("fastsafetensors")
