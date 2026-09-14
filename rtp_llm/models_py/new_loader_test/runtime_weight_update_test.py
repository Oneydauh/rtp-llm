import unittest
from unittest import mock

import torch

from rtp_llm.models_py.runtime_weight_update import RuntimeWeightTransaction, TensorUpdate


class RuntimeWeightTransactionTest(unittest.TestCase):
    def test_commit_preserves_parameter_and_exported_transpose(self):
        parameter = torch.nn.Parameter(torch.arange(12.0).reshape(3, 4).t(), requires_grad=False)
        exported = parameter.data.t()
        pointer, stride, identity = parameter.data_ptr(), parameter.stride(), id(parameter)
        tx = RuntimeWeightTransaction([TensorUpdate("weight", parameter, torch.full_like(parameter, 7))])
        tx.install()
        tx.commit()
        self.assertEqual((parameter.data_ptr(), parameter.stride(), id(parameter)), (pointer, stride, identity))
        torch.testing.assert_close(exported, torch.full_like(exported, 7))

    def test_invalid_last_tensor_does_not_modify_first(self):
        first, second = torch.ones(2), torch.ones(3)
        with self.assertRaisesRegex(ValueError, "mismatch"):
            RuntimeWeightTransaction([
                TensorUpdate("first", first, torch.zeros(2)),
                TensorUpdate("second", second, torch.zeros(4)),
            ])
        torch.testing.assert_close(first, torch.ones(2))

    def test_rollback_after_rank_failure_restores_all_weights(self):
        weight, scale = torch.ones(2), torch.full((1,), 2.0)
        tx = RuntimeWeightTransaction([
            TensorUpdate("weight", weight, torch.full((2,), 3.0)),
            TensorUpdate("scale", scale, torch.full((1,), 4.0)),
        ])
        tx.install()
        tx.rollback()
        tx.rollback()
        torch.testing.assert_close(weight, torch.ones(2))
        torch.testing.assert_close(scale, torch.full((1,), 2.0))
        with self.assertRaisesRegex(RuntimeError, "rolled_back"):
            tx.install()

    def test_install_failure_restores_previously_copied_tensor(self):
        first, second = torch.ones(2), torch.ones(2)
        tx = RuntimeWeightTransaction([
            TensorUpdate("first", first, torch.full((2,), 3.0)),
            TensorUpdate("second", second, torch.full((2,), 4.0)),
        ])
        copy = torch.Tensor.copy_
        calls = 0

        def fail_once(target, source, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("injected copy failure")
            return copy(target, source, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "copy_", fail_once):
            with self.assertRaisesRegex(RuntimeError, "injected"):
                tx.install()
        torch.testing.assert_close(first, torch.ones(2))
        torch.testing.assert_close(second, torch.ones(2))

    def test_source_snapshot_survives_ipc_sender_mutation_and_cross_alias(self):
        first, second = torch.ones(2), torch.full((2,), 2.0)
        tx = RuntimeWeightTransaction([
            TensorUpdate("first", first, second), TensorUpdate("second", second, first)
        ])
        tx.install()
        tx.commit()
        torch.testing.assert_close(first, torch.full((2,), 2.0))
        torch.testing.assert_close(second, torch.ones(2))

    def test_aliased_targets_require_single_owner(self):
        weight = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "single owner"):
            RuntimeWeightTransaction([
                TensorUpdate("weight", weight, torch.zeros(4)),
                TensorUpdate("alias", weight.view(2, 2), torch.zeros(2, 2)),
            ])

    def test_rejects_dtype_cast_and_overlapping_target(self):
        with self.assertRaisesRegex(ValueError, "mismatch"):
            RuntimeWeightTransaction([TensorUpdate("weight", torch.ones(2), torch.ones(2).half())])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            RuntimeWeightTransaction([TensorUpdate("weight", torch.ones(1).expand(3), torch.zeros(3))])

    def test_rejects_storage_replacement_after_prepare(self):
        weight = torch.nn.Parameter(torch.ones(2), requires_grad=False)
        tx = RuntimeWeightTransaction([TensorUpdate("weight", weight, torch.zeros(2))])
        weight.data = torch.full((2,), 8.0)
        with self.assertRaisesRegex(RuntimeError, "storage changed"):
            tx.install()
        torch.testing.assert_close(weight, torch.full((2,), 8.0))

    def test_float8_preserves_dtype_and_bytes(self):
        weight = torch.ones(4).to(torch.float8_e4m3fn)
        tx = RuntimeWeightTransaction([TensorUpdate("weight", weight, torch.full((4,), 2.0).to(weight.dtype))])
        tx.install()
        tx.rollback()
        torch.testing.assert_close(weight.float(), torch.ones(4))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/ROCm")
    def test_cuda_graph_observes_update_and_rollback_without_recapture(self):
        weight = torch.ones(4, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = weight * 2
        tx = RuntimeWeightTransaction([TensorUpdate("weight", weight, torch.full((4,), 3.0))])
        tx.install()
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, 6))
        tx.rollback()
        graph.replay()
        torch.testing.assert_close(output, torch.full_like(output, 2))


if __name__ == "__main__":
    unittest.main()
