"""Qwen3-VL multimodal front-end (mm_part) for the new-loader path.

Reuses Qwen2-VL's mm_part machinery (image_processor, mm_embedding interface,
non-interleaved mrope ``get_position_ids``) but plugs in the *already loaded*
new-loader vision tower (``py_model.visual``) instead of constructing/loading a
fresh ViT.

Two Qwen3-VL specifics are deliberately deferred (step 2):
  * interleaved M-RoPE — ``get_position_ids`` here is Qwen2-VL's non-interleaved
    variant, so image-token positions are approximate (text is exact).
  * DeepStack — ``py_model.visual`` returns ``(embeds, deepstack_features)``;
    the adapter keeps only ``embeds`` and stashes the deepstack features on
    ``last_deepstack`` for a future injection path.
"""

from typing import Any

import torch
import torch.nn as nn

from rtp_llm.models.qwen2_vl.qwen2_vl_vit import (
    Qwen2VLImageEmbedding,
    Qwen2VLImageProcessor,
)


class _VisualEmbedsAdapter(nn.Module):
    """Wraps the new-loader ``Qwen3VLVisionTransformer`` so it matches the
    contract Qwen2VLImageEmbedding expects: ``visual(pixel_values,
    grid_thw=...) -> Tensor`` plus ``get_device()``.
    """

    def __init__(self, visual: nn.Module):
        super().__init__()
        self.inner = visual
        self.last_deepstack: Any = None

    def get_device(self) -> torch.device:
        return next(self.inner.parameters()).device

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        out = self.inner(pixel_values, grid_thw=grid_thw)
        if isinstance(out, (tuple, list)):
            self.last_deepstack = out[1] if len(out) > 1 else None
            return out[0]
        return out


class Qwen3VLImageEmbedding(Qwen2VLImageEmbedding):
    def __init__(
        self,
        mm_related_params: Any,
        visual: nn.Module,
        model_config: Any = None,
    ):
        # NOTE: intentionally does NOT call super().__init__ — that would build
        # a fresh (random-weight) ViT. We reuse the already-loaded one.
        self.mm_related_params = mm_related_params
        self.config = model_config
        self.image_processor = Qwen2VLImageProcessor.from_pretrained(
            mm_related_params.config["ckpt_path"]
        )
        self.visual = _VisualEmbedsAdapter(visual)
