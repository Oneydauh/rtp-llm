import unittest

import torch
from rtp_llm.models_py.layers.embedding import ParallelLMHead, VocabParallelEmbedding
from rtp_llm.models_py.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from rtp_llm.models_py.read_spec import LoadedCheckpointTensor


class ParallelLayerReadSpecTest(unittest.TestCase):
    def test_column_parallel_accepts_reader_shard(self):
        layer = ColumnParallelLinear(
            4,
            6,
            tp_size=2,
            tp_rank=1,
            bias=True,
            params_dtype=torch.float32,
            prefix="column",
        )
        weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        bias = torch.arange(6, dtype=torch.float32)

        weight_spec = layer.checkpoint_read_spec("weight", tuple(weight.shape))
        bias_spec = layer.checkpoint_read_spec("bias", tuple(bias.shape))
        self.assertEqual(weight_spec.shard_dim, 0)
        self.assertEqual(bias_spec.shard_dim, 0)
        layer.load_weights(
            {
                "weight": weight_spec.materialize(weight),
                "bias": bias_spec.materialize(bias),
            }
        )

        self.assertTrue(torch.equal(layer.weight, weight[3:]))
        self.assertTrue(torch.equal(layer.bias, bias[3:]))

    def test_column_parallel_transposed_checkpoint_shards_source_dim(self):
        layer = ColumnParallelLinear(
            4,
            6,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="column",
        )
        canonical = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        transposed = canonical.t().contiguous()
        read_spec = layer.checkpoint_read_spec("weight", tuple(transposed.shape))

        self.assertEqual(read_spec.shard_dim, 1)
        layer.load_weights({"weight": read_spec.materialize(transposed)})
        self.assertTrue(torch.equal(layer.weight, canonical[3:]))

    def test_row_parallel_accepts_reader_shard(self):
        layer = RowParallelLinear(
            6,
            4,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="row",
        )
        weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        read_spec = layer.checkpoint_read_spec("weight", tuple(weight.shape))

        self.assertEqual(read_spec.shard_dim, 1)
        local = read_spec.materialize(weight)
        self.assertIsInstance(local, LoadedCheckpointTensor)
        layer.load_weights({"weight": local})
        self.assertTrue(torch.equal(layer.weight, weight[:, 3:]))

    def test_row_parallel_transposed_checkpoint_shards_source_dim(self):
        layer = RowParallelLinear(
            6,
            4,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="row",
        )
        canonical = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        transposed = canonical.t().contiguous()
        read_spec = layer.checkpoint_read_spec("weight", tuple(transposed.shape))

        self.assertEqual(read_spec.shard_dim, 0)
        layer.load_weights({"weight": read_spec.materialize(transposed)})
        self.assertTrue(torch.equal(layer.weight, canonical[:, 3:]))

    def test_vocab_embedding_and_lm_head_accept_reader_shards(self):
        embedding = VocabParallelEmbedding(
            8, 4, tp_size=2, tp_rank=1, params_dtype=torch.float32
        )
        lm_head = ParallelLMHead(8, 4, tp_size=2, tp_rank=1, params_dtype=torch.float32)
        weight = torch.arange(32, dtype=torch.float32).reshape(8, 4)

        embedding_spec = embedding.checkpoint_read_spec("weight", tuple(weight.shape))
        lm_head_spec = lm_head.checkpoint_read_spec("weight", tuple(weight.shape))
        embedding.load_weights({"weight": embedding_spec.materialize(weight)})
        lm_head.load_weights({"weight": lm_head_spec.materialize(weight)})

        self.assertTrue(torch.equal(embedding.weight, weight[4:]))
        self.assertTrue(torch.equal(lm_head.weight, weight[4:]))

    def test_merged_column_per_projection_keys_accept_reader_shards(self):
        layer = MergedColumnParallelLinear(
            4,
            12,
            tp_size=2,
            tp_rank=1,
            shard_names=["gate_proj", "up_proj"],
            params_dtype=torch.float32,
            prefix="merged",
        )
        gate = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        up = torch.arange(24, 48, dtype=torch.float32).reshape(6, 4)
        gate_spec = layer.checkpoint_read_spec("gate_proj.weight", tuple(gate.shape))
        up_spec = layer.checkpoint_read_spec("up_proj.weight", tuple(up.shape))

        layer.load_weights(
            {
                "gate_proj.weight": gate_spec.materialize(gate),
                "up_proj.weight": up_spec.materialize(up),
            }
        )
        self.assertTrue(torch.equal(layer.weight[:3], gate[3:]))
        self.assertTrue(torch.equal(layer.weight[3:], up[3:]))

    def test_qkv_per_projection_keys_accept_reader_shards(self):
        layer = QKVParallelLinear(
            hidden_size=4,
            num_heads=4,
            num_kv_heads=2,
            head_dim=1,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="qkv",
        )
        q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        k = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        v = torch.arange(8, 16, dtype=torch.float32).reshape(2, 4)
        q_spec = layer.checkpoint_read_spec("q_proj.weight", tuple(q.shape))
        k_spec = layer.checkpoint_read_spec("k_proj.weight", tuple(k.shape))
        v_spec = layer.checkpoint_read_spec("v_proj.weight", tuple(v.shape))

        layer.load_weights(
            {
                "q_proj.weight": q_spec.materialize(q),
                "k_proj.weight": k_spec.materialize(k),
                "v_proj.weight": v_spec.materialize(v),
            }
        )
        expected = torch.cat((q[2:], k[1:], v[1:]), dim=0)
        self.assertTrue(torch.equal(layer.weight, expected))

    def test_existing_full_tensor_path_remains_supported(self):
        column = ColumnParallelLinear(
            4,
            6,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="column",
        )
        row = RowParallelLinear(
            6,
            4,
            tp_size=2,
            tp_rank=1,
            params_dtype=torch.float32,
            prefix="row",
        )
        column_weight = torch.arange(24, dtype=torch.float32).reshape(6, 4)
        row_weight = torch.arange(24, dtype=torch.float32).reshape(4, 6)

        column.load_weights({"weight": column_weight})
        row.load_weights({"weight": row_weight})

        self.assertTrue(torch.equal(column.weight, column_weight[3:]))
        self.assertTrue(torch.equal(row.weight, row_weight[:, 3:]))


if __name__ == "__main__":
    unittest.main()
