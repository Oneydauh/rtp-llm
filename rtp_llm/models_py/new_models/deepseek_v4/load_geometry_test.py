import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from rtp_llm.models_py.model_loader import NewLoaderConfig
from rtp_llm.models_py.new_models.deepseek_v4.load_geometry import (
    validate_load_groups,
    validate_v4_static_placement,
)


class FlashLoadGeometryTest(unittest.TestCase):
    def config(self, **kwargs):
        return NewLoaderConfig(
            device="cpu",
            compute_dtype=torch.bfloat16,
            tp_size=2,
            tp_rank=1,
            **kwargs,
        )

    def test_cp2_replicates_attention_without_replicating_experts(self):
        native = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2, ep_rank=1)
        validate_load_groups(
            native, self.config(attn_tp_size=1, attn_tp_rank=0, ep_size=2, ep_rank=1)
        )

    def test_physical_tp_may_not_overwrite_cp_group(self):
        native = SimpleNamespace(tp_size=1, tp_rank=0, ep_size=2, ep_rank=1)
        with self.assertRaisesRegex(ValueError, "attention rank mismatch"):
            validate_load_groups(native, self.config(ep_size=2, ep_rank=1))

    def test_wrong_expert_rank_rejected_even_at_equal_world_size(self):
        native = SimpleNamespace(tp_size=2, tp_rank=1, ep_size=2, ep_rank=1)
        with self.assertRaisesRegex(ValueError, "expert rank mismatch"):
            validate_load_groups(native, self.config(ep_size=2, ep_rank=0))

    def test_nonidentity_and_redundancy_are_never_contiguous_ownership(self):
        native = SimpleNamespace(
            n_layers=2, n_routed_experts=4, n_physical_experts=4, commit_only=False
        )
        load = self.config(ep_size=2, ep_rank=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "phy2log.json"
            model = SimpleNamespace(num_layers=2, phy2log_path=path)
            identity = [0, 1, 2, 3]
            path.write_text(json.dumps([identity, identity]))
            validate_v4_static_placement(native, model, load)
            path.write_text(json.dumps([identity, [2, 1, 0, 3]], indent=2))
            with self.assertRaisesRegex(ValueError, "layer 1 requires identity"):
                validate_v4_static_placement(native, model, load)
            native.n_physical_experts = 6
            with self.assertRaisesRegex(ValueError, "redundant experts"):
                validate_v4_static_placement(native, model, load)
            native.commit_only = True
            validate_v4_static_placement(native, model, load)


if __name__ == "__main__":
    unittest.main()
