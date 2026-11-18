"""Direct parameter ownership for Qwen classifier and reranker checkpoints."""

import torch
from torch import nn

from rtp_llm.models_py.module_base import _mark_loaded
from rtp_llm.models_py.read_spec import unwrap_checkpoint_tensor


def handler_weight_mappings(names):
    """Turn declared checkpoint names into the embedding engine's global export roles."""
    if not isinstance(names, tuple) or not all(
        isinstance(name, str) and name for name in names
    ):
        raise TypeError("Handler checkpoint names must be a tuple of nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("Duplicate handler checkpoint name")
    return tuple(("__custom__." + name, name) for name in names)


class ReplicatedHandlerHeads(nn.Module):
    """Handler projections consume full, unprocessed weights on each serving rank."""

    # Unlike the classifier head, these are shared *checkpoint inputs* with the
    # language model: its TP/PP slice, pruning, or logit scaling is not valid
    # as the handler's full-vocabulary output projection.
    vocabulary_sources = frozenset(("embed_tokens.weight", "lm_head.weight"))
    allowed = vocabulary_sources | {"classifier.weight", "classifier.bias"}

    def __init__(self, mappings, mapper, hidden_size, dtype):
        super().__init__()
        self.hidden_size = hidden_size
        self.sources = {}
        for target, source in mappings:
            name = mapper.map_name(source)
            if name not in self.allowed or name in self.sources:
                raise ValueError(
                    f"Unsupported or ambiguous Qwen handler source {source}"
                )
            self.sources[name] = (str(len(self.sources)), target)
        self.params = nn.ParameterDict()
        self.register_buffer("_anchor", torch.empty(0, dtype=dtype), persistent=False)

    def capture(self, name, value):
        """Return original tensor for optional dispatch into the language owner."""
        key, _ = self.sources[name]
        tensor, spec = unwrap_checkpoint_tensor(value)
        if key in self.params:
            raise ValueError(f"Duplicate downstream checkpoint tensor {name}")
        if spec is not None and (
            spec.slices is not None or tuple(tensor.shape) != spec.source_shape
        ):
            raise ValueError(f"Downstream checkpoint tensor {name} must be replicated")
        is_bias = name == "classifier.bias"
        if (
            not tensor.is_floating_point()
            or tensor.ndim != (1 if is_bias else 2)
            or tensor.shape[0] < 1
            or (not is_bias and tensor.shape[1] != self.hidden_size)
        ):
            raise ValueError(f"Invalid downstream tensor {name}: {tuple(tensor.shape)}")
        # FST frees its backing storage; the handler owns its registered parameter.
        self.params[key] = nn.Parameter(
            tensor.detach().to(self._anchor, copy=True), requires_grad=False
        )
        _mark_loaded(self.params, key)
        return tensor

    def validate(self):
        missing = [
            name for name, (key, _) in self.sources.items() if key not in self.params
        ]
        if missing:
            raise RuntimeError(f"Missing downstream checkpoint tensors: {missing}")
        if {"classifier.weight", "classifier.bias"} <= self.sources.keys():
            w_key, _ = self.sources["classifier.weight"]
            b_key, _ = self.sources["classifier.bias"]
            if self.params[w_key].shape[0] != self.params[b_key].shape[0]:
                raise ValueError("Classifier weight and bias label counts disagree")

    def exports(self):
        self.validate()
        return {target: self.params[key] for key, target in self.sources.values()}
