"""Bias-free HF Llama checkpoint owner, including compatible Mistral HF weights."""

import json
from pathlib import Path

from rtp_llm.models_py.new_models.qwen2.language import Qwen2ForCausalLM


class LlamaForCausalLM(Qwen2ForCausalLM):
    # The HF Q/K rotary channel ordering, gated MLP and TP slicing are the
    # same as Qwen2. Do not create (or falsely require) a packed QKV bias.
    QKV_BIAS = False

    def __init__(self, model_config, load_config):
        get = (
            model_config.get
            if isinstance(model_config, dict)
            else lambda name, default=None: getattr(model_config, name, default)
        )
        path = get("ckpt_path")
        hf = model_config if isinstance(model_config, dict) else {}
        if path:
            with (Path(path) / "config.json").open() as stream:
                hf = json.load(stream)
        family = hf.get("model_type", "llama")
        expected = {"llama": "LlamaForCausalLM", "mistral": "MistralForCausalLM"}
        if family not in expected or expected[family] not in hf.get(
            "architectures", [expected.get(family)]
        ):
            raise ValueError(f"Unsupported HF decoder for llama checkpoint: {family!r}")
        if (
            hf.get("attention_bias")
            or hf.get("mlp_bias")
            or hf.get("hidden_act", "silu") != "silu"
        ):
            raise ValueError("The llama checkpoint owner requires bias-free SwiGLU")
        # This is the explicitly requested llama route, not a Mistral runtime
        # registration. Full and sliding causal attention agree only within
        # one window; never silently run full attention past that point.
        window = hf.get("sliding_window")
        if window is not None and not 0 < get("max_seq_len", 0) <= window:
            raise ValueError("The llama route requires max_seq_len <= sliding_window")
        super().__init__(model_config, load_config)
