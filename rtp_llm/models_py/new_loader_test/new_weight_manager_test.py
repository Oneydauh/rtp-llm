import threading
import unittest
from unittest import mock

import torch

from rtp_llm.model_loader.new_weight_manager import NewLoaderWeightManager
from rtp_llm.models_py.model_loader import NewModelLoader
from rtp_llm.models_py.new_models.qwen3.test import test_qwen3_dense_load as fixtures


class NewWeightManagerTest(unittest.TestCase):
    def make_model(self, rank=0, tp=1):
        fixture = fixtures.Qwen3LoadTest()
        config, load_config = fixture._config(), fixture._load_config(tp_size=tp, tp_rank=rank)
        model = fixtures.Qwen3ForCausalLM(config, load_config)
        weights = fixture._weights()
        model.load_weights(weights)
        NewModelLoader._validate_loaded_weights(model)
        model.process_weights_after_loading()
        loader = NewModelLoader(config, load_config)
        return model, loader, weights

    def test_repeated_tp_partial_fused_updates_match_fresh_checkpoint_load(self):
        for rank in (0, 1):
            model, loader, weights = self.make_model(rank, 2)
            manager = NewLoaderWeightManager(model, loader)
            before = {name: (id(t), t.data_ptr(), t.stride()) for name, t in model.named_parameters()}
            changes = {
                "model.layers.0.self_attn.q_proj.weight": torch.full((4, 4), 9.0),
                "model.layers.0.self_attn.k_proj.weight": torch.full((2, 4), 8.0),
                "model.layers.0.mlp.gate_proj.weight": torch.full((4, 4), 7.0),
                "model.layers.0.mlp.up_proj.weight": torch.full((4, 4), 6.0),
                "model.layers.0.mlp.down_proj.weight": torch.full((4, 4), 5.0),
            }
            for name, value in changes.items():
                manager.update_tensors({name: value})
            fresh = fixtures.Qwen3ForCausalLM(loader.model_config, loader.load_config)
            fresh.load_weights({**weights, **changes})
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter, dict(fresh.named_parameters())[name])
                self.assertEqual((id(parameter), parameter.data_ptr(), parameter.stride()), before[name])
            self.assertEqual(manager.version, len(changes))

    def test_tied_embedding_update_reaches_cpp_views_and_lm_head_postprocess_once(self):
        model, loader, _ = self.make_model(1, 2)
        model.normalize_lm_head_weight = True
        model.logit_scale = 2.0
        exported = model.runtime_weight_view()
        manager = NewLoaderWeightManager(model, loader)
        embedding = torch.arange(32.0).reshape(8, 4) + 1
        for _ in range(2):
            manager.update_tensors({"model.embed_tokens.weight": embedding})
            torch.testing.assert_close(exported["embedding"], embedding[4:])
            torch.testing.assert_close(exported["lm_head"], torch.nn.functional.normalize(embedding[4:], dim=1) * 2)
        with self.assertRaisesRegex(ValueError, "non-owning"):
            manager.update_tensors({"lm_head.weight": embedding})

    def test_failed_batch_leaves_all_parameters_and_version_unchanged(self):
        model, loader, _ = self.make_model()
        manager = NewLoaderWeightManager(model, loader)
        before = {name: value.clone() for name, value in model.named_parameters()}
        with self.assertRaises(ValueError):
            manager.update_tensors({
                "model.norm.weight": torch.zeros(4),
                "model.layers.0.mlp.down_proj.weight": torch.ones(7, 7),
            })
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, before[name])
        self.assertEqual(manager.version, 0)

    def test_storage_replacement_fails_before_next_update(self):
        model, loader, _ = self.make_model()
        manager = NewLoaderWeightManager(model, loader)
        model.norm.weight = torch.nn.Parameter(torch.ones(4), requires_grad=False)
        with self.assertRaisesRegex(RuntimeError, "parameter/storage changed"):
            manager.update_tensors({"model.norm.weight": torch.zeros(4)})
        self.assertEqual(manager.version, 0)

    def test_forward_lock_excludes_update_writer(self):
        model, loader, _ = self.make_model()
        manager = NewLoaderWeightManager(model, loader)
        started, finished = threading.Event(), threading.Event()

        def update():
            started.set()
            manager.update_tensors({"model.norm.weight": torch.zeros(4)})
            finished.set()

        with model._weight_update_lock:
            thread = threading.Thread(target=update)
            thread.start()
            self.assertTrue(started.wait(2))
            self.assertFalse(finished.wait(0.05))
            torch.testing.assert_close(model.norm.weight, torch.ones(4))
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(finished.is_set())
        torch.testing.assert_close(model.norm.weight, torch.zeros(4))

    def test_ipc_request_dispatches_global_checkpoint_tensor(self):
        model, loader, _ = self.make_model()
        manager = NewLoaderWeightManager(model, loader)
        with mock.patch("rtp_llm.model_loader.tipc.SharedMemIpcMeta.decode", return_value=mock.sentinel.meta), mock.patch(
            "rtp_llm.model_loader.tipc.SharedMemoryIPCHelper.build_from_meta", return_value=torch.zeros(4)
        ):
            manager.update({"method": "shm", "name": "model.norm.weight", "desc": "test"})
        torch.testing.assert_close(model.norm.weight, torch.zeros(4))
        with self.assertRaisesRegex(ValueError, "transport"):
            manager.update({"method": "bad", "name": "model.norm.weight", "desc": "test"})


if __name__ == "__main__":
    unittest.main()
