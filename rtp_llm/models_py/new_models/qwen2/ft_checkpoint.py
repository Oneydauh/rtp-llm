"""Qwen2/Llama's boundary conversion for legacy dense FT runtime snapshots.

Historical names exist only at this import boundary. No runtime weight-name
registry or OldLoader/device conversion dependency is introduced here.
"""

import torch

from rtp_llm.models_py.ft_checkpoint import RuntimeTensorBinding
from rtp_llm.models_py.quant_methods.unquantized import UnquantizedLinearMethod


def _identity(values):
    return values[0]


def _transpose(values):
    if values[0].ndim != 2:
        raise ValueError("Dense FT kernel must be a two-dimensional matrix")
    return values[0].T


def _bias(values):
    value = values[0]
    if value.ndim != 1 and not (value.ndim == 2 and value.shape[0] == 1):
        raise ValueError("Dense FT QKV bias must have shape [N] or [1, N]")
    return value.reshape(-1)


def load_dense_ft_checkpoint(model, checkpoint):
    from rtp_llm.models_py.quant_methods.fp8 import (
        Fp8BlockLinearMethod,
        Fp8LinearMethod,
        Fp8PerChannelLinearMethod,
    )

    fp8_methods = (Fp8LinearMethod, Fp8PerChannelLinearMethod, Fp8BlockLinearMethod)

    if type(model).__name__ not in ("Qwen2ForCausalLM", "LlamaForCausalLM"):
        raise ValueError("FT adapter is only validated for dense Qwen2/Llama")
    swizzle_flags = set()
    for key, value in checkpoint.env_params.items():
        normalized = key.lower().replace("_", "")
        if normalized == "useswizzlea":
            if str(value).lower() not in ("0", "1", "false", "true", ""):
                raise ValueError("FT use_swizzleA metadata must be a boolean")
            swizzle_flags.add(str(value).lower() in ("1", "true"))
        if normalized == "armgemmusekai" and str(value).lower() not in (
            "0",
            "false",
            "none",
            "",
        ):
            raise ValueError("Historical ARM Kai FT layout has no current-main decoder")
    if len(swizzle_flags) > 1:
        raise ValueError("Conflicting FT use_swizzleA metadata")
    swizzled = True in swizzle_flags

    def kernel(values):
        value = _transpose(values)
        if swizzled:
            from rtp_llm.models_py.quant_methods.dense_layout import decode_dense_weight

            value = decode_dense_weight(value)
        return value

    for layer in model.layers:
        for linear in (
            layer.self_attn.qkv_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_up_proj,
            layer.mlp.down_proj,
        ):
            if not isinstance(
                linear.quant_method, (UnquantizedLinearMethod, *fp8_methods)
            ):
                raise ValueError(
                    "Dense FT has no declared adapter for this quantized runtime layout"
                )
            if isinstance(linear.quant_method, fp8_methods) and swizzled:
                raise ValueError(
                    "FP8 FT swizzled source requires an explicit quantized layout decoder"
                )
    if getattr(model.lm_head, "output_vocab_ids", None):
        raise ValueError(
            "Legacy FT checkpoints cannot be combined with vocabulary pruning"
        )

    tp = checkpoint.topology.tp_size
    rank = checkpoint.topology.tp_rank

    def fp8_bindings(linear, source, destination, projection, layer_prefix):
        from rtp_llm.models_py.quant_methods.fp8_ft import (
            PerChannelFP8FTProjection,
            PerTensorFP8FTProjection,
        )

        scale = source.removesuffix(".kernel") + ".weight_only_quant_scale"
        if isinstance(linear.quant_method, Fp8BlockLinearMethod):
            from rtp_llm.models_py.quant_methods.block_fp8_ft import block_ft_bindings

            return block_ft_bindings(linear, source, scale, destination, checkpoint)
        if isinstance(linear.quant_method, Fp8PerChannelLinearMethod):
            state = PerChannelFP8FTProjection(
                linear, checkpoint, source, scale, projection
            )
            return [
                RuntimeTensorBinding(
                    (source, scale),
                    destination + ".weight",
                    _identity,
                    lambda: (state.weight(),),
                ),
                RuntimeTensorBinding(
                    (source, scale),
                    destination + ".weight_scale",
                    _identity,
                    lambda: (state.scale(),),
                ),
            ]
        state = PerTensorFP8FTProjection(linear, checkpoint, source, scale, projection)
        # These spellings are only legacy serializer boundary names. The quant
        # module owns all tensor layout, fusion and activation-scale semantics.
        stems = {
            "qkv": "pre_layernorm_weights",
            "out": "self_attention_weights.attention_output_weight",
            "gate_up": "post_layernorm_weights",
            "down": "ffn_weights.intermediate_weight2",
        }
        stem = layer_prefix + stems[projection]
        direct, reciprocal = stem + ".static_quant", stem + ".static_quant_reciprocal"
        dynamic = linear.quant_method.quant_config.activation_dynamic
        if dynamic and (
            direct in checkpoint.entries or reciprocal in checkpoint.entries
        ):
            raise ValueError(
                "FP8 FT static source disagrees with dynamic activation configuration"
            )
        if not dynamic and not {direct, reciprocal}.issubset(checkpoint.entries):
            raise ValueError("FP8 FT static activation metadata is incomplete")
        actual, companion = (
            (direct, reciprocal) if projection == "out" else (reciprocal, direct)
        )
        auxiliary = () if dynamic else (actual, companion)
        return [
            RuntimeTensorBinding(
                (source, scale),
                destination + ".weight",
                _identity,
                lambda: (state.weight(),),
            ),
            RuntimeTensorBinding(
                (scale,),
                destination + ".weight_scale",
                _identity,
                lambda: (state.scale(),),
            ),
            RuntimeTensorBinding(
                (scale, *auxiliary),
                destination + ".input_scale",
                _identity,
                lambda: (
                    state.input_scale(
                        None if dynamic else actual, companion, projection != "down"
                    ),
                ),
            ),
        ]

    embedding = model.embed_tokens
    head = model.lm_head
    if any(
        (leaf.tp_size, leaf.tp_rank) != (tp, rank)
        for leaf in (embedding, head)
        if leaf is not None
    ):
        raise ValueError("FT import requires matching embedding/head TP topology")

    def embedding_rows(_):
        # OldLoader shards embedding on hidden, NewLoader on vocabulary. Read
        # only this rank's destination vocabulary rows from each peer shard.
        rows, hidden = embedding.vocab_size, embedding.embedding_dim
        chunks = []
        for peer in range(tp):
            if checkpoint.read_rank_slice("global.embedding", peer) != (
                rows,
                hidden // tp,
            ):
                raise ValueError("FT embedding hidden-partition layout mismatch")
            chunks.append(
                checkpoint.read_rank_slice(
                    "global.embedding",
                    peer,
                    (
                        slice(embedding.vocab_start_idx, embedding.vocab_end_idx),
                        slice(None),
                    ),
                )
            )
        return embedding.prepare_local_checkpoint_weight(torch.cat(chunks, dim=1))

    def head_rows(_):
        # sp_0_pad8 partitions padded vocabulary, which changes rank boundaries.
        # A tied source may have more rows than the configured output vocabulary.
        # Read the source's actual shard stride, not the destination's padding.
        first_shape = checkpoint.read_rank_slice("global.lm_head", 0)
        if len(first_shape) != 2 or first_shape[1] != head.hidden_size:
            raise ValueError("FT lm_head padded-vocabulary layout mismatch")
        old_rows, hidden = first_shape
        if old_rows * tp < head.checkpoint_vocab_size:
            raise ValueError(
                "FT lm_head does not cover the configured output vocabulary"
            )
        begin = min(rank * head.vocab_size_per_partition, head.checkpoint_vocab_size)
        end = begin + len(head.local_checkpoint_rows)
        chunks = []
        for peer in range(tp):
            if checkpoint.read_rank_slice("global.lm_head", peer) != (old_rows, hidden):
                raise ValueError("FT lm_head padded-vocabulary layout mismatch")
            left, right = max(begin, peer * old_rows), min(end, (peer + 1) * old_rows)
            if left < right:
                chunks.append(
                    checkpoint.read_rank_slice(
                        "global.lm_head",
                        peer,
                        (
                            slice(left - peer * old_rows, right - peer * old_rows),
                            slice(None),
                        ),
                    )
                )
        selected = (
            torch.cat(chunks, dim=0)
            if chunks
            else head.weight.detach().new_empty((0, hidden))
        )
        return head.prepare_local_checkpoint_weight(selected)

    def ffn_local(names, projection, local_width, hidden):
        # Main pads intermediate_size to TP*64 when swizzleA is enabled.
        # Consequently the original FT rank boundaries differ from the model's
        # unpadded local parameter boundaries, just as with padded vocabulary.
        width = ((local_width + 63) // 64) * 64 if swizzled else local_width
        global_width = local_width * tp
        begin, end = rank * local_width, (rank + 1) * local_width
        parts = [[], []] if projection == "gate_up" else [[]]
        for peer in range(tp):
            left, right = max(begin, peer * width), min(end, (peer + 1) * width)
            padding = max(0, global_width - peer * width)
            needs_padding_check = width != local_width and padding < width
            expected = (
                (width, hidden)
                if projection == "down"
                else (hidden, width * (2 if len(names) == 1 else 1))
            )
            for name in names:
                if checkpoint.read_rank_slice(name, peer) != expected:
                    raise ValueError("FT FFN padded-partition layout mismatch")
            if left >= right and not needs_padding_check:
                continue
            raw = tuple(
                checkpoint.read_rank_slice(name, peer, (slice(None), slice(None)))
                for name in names
            )
            if projection == "down":
                canonical = (kernel(raw),)
                axis = 1
            elif len(names) == 1:
                canonical = kernel(raw).split(width, dim=0)
                axis = 0
            else:
                # ROCm's rewrite list includes separate w3, not w1.
                canonical = (_transpose(raw[:1]), kernel(raw[1:]))
                axis = 0
            for output, value in zip(parts, canonical):
                if (
                    needs_padding_check
                    and torch.count_nonzero(
                        value.narrow(axis, padding, width - padding)
                    ).item()
                ):
                    raise ValueError("FT FFN padding contains nonzero values")
                if left < right:
                    output.append(value.narrow(axis, left - peer * width, right - left))
        merged = [
            torch.cat(values, dim=1 if projection == "down" else 0) for values in parts
        ]
        return (merged[0] if projection == "down" else torch.cat(merged, dim=0),)

    bindings = []
    if embedding is not None:
        bindings.append(
            RuntimeTensorBinding(
                ("global.embedding",), "embed_tokens.weight", embedding_rows
            )
        )
    if head is not None:
        bindings.append(
            RuntimeTensorBinding(("global.lm_head",), "lm_head.weight", head_rows)
        )
    if model.norm is not None:
        bindings.append(
            RuntimeTensorBinding(
                ("global.final_layernorm.gamma",), "norm.weight", _identity
            )
        )
    validators = {}
    if "global.final_layernorm.beta" in checkpoint.entries:

        def zero_final_beta(value):
            if (
                value.shape != model.norm.weight.shape
                or value.dtype != model.norm.weight.dtype
                or torch.count_nonzero(value).item()
            ):
                raise ValueError("Dense RMSNorm FT final beta must be exactly zero")

        validators["global.final_layernorm.beta"] = zero_final_beta
    layer_ids = (
        model.layer_ids() if hasattr(model, "layer_ids") else range(len(model.layers))
    )
    if len(layer_ids) != len(model.layers):
        raise ValueError("FT pipeline range disagrees with local decoder layers")
    for index, (layer_id, layer) in enumerate(zip(layer_ids, model.layers)):
        src = f"layers.{layer_id}."
        dst = f"layers.{index}."
        for old, target, convert in (
            ("pre_layernorm_weights.gamma", "input_layernorm.weight", _identity),
            (
                "post_layernorm_weights.gamma",
                "post_attention_layernorm.weight",
                _identity,
            ),
            (
                "self_attention_weights.query_weight.kernel",
                "self_attn.qkv_proj.weight",
                kernel,
            ),
            (
                "self_attention_weights.attention_output_weight.kernel",
                "self_attn.o_proj.weight",
                kernel,
            ),
        ):
            if target in ("self_attn.qkv_proj.weight", "self_attn.o_proj.weight"):
                linear = (
                    layer.self_attn.qkv_proj
                    if "qkv_proj" in target
                    else layer.self_attn.o_proj
                )
                if isinstance(linear.quant_method, fp8_methods):
                    bindings.extend(
                        fp8_bindings(
                            linear,
                            src + old,
                            dst + target.removesuffix(".weight"),
                            "qkv" if "qkv_proj" in target else "out",
                            src,
                        )
                    )
                    continue
            bindings.append(RuntimeTensorBinding((src + old,), dst + target, convert))
        if layer.self_attn.qkv_proj.bias is not None:
            bindings.append(
                RuntimeTensorBinding(
                    (src + "self_attention_weights.query_weight.bias",),
                    dst + "self_attn.qkv_proj.bias",
                    _bias,
                )
            )
        down_names = (src + "ffn_weights.intermediate_weight2.kernel",)
        gate_names = (
            (src + "ffn_weights.intermediate_weight13.kernel",)
            if src + "ffn_weights.intermediate_weight13.kernel" in checkpoint.entries
            else (
                src + "ffn_weights.intermediate_weight.kernel",
                src + "ffn_weights.intermediate_weight3.kernel",
            )
        )
        local_width = layer.mlp.down_proj.weight.shape[1]
        hidden = layer.mlp.down_proj.weight.shape[0]
        for names, projection, target in (
            (down_names, "down", "mlp.down_proj.weight"),
            (gate_names, "gate_up", "mlp.gate_up_proj.weight"),
        ):
            linear = (
                layer.mlp.down_proj if projection == "down" else layer.mlp.gate_up_proj
            )
            if isinstance(linear.quant_method, fp8_methods):
                if len(names) != 1:
                    raise ValueError(
                        "Per-tensor FP8 FT requires the producer's fused gate/up tensor"
                    )
                bindings.extend(
                    fp8_bindings(
                        linear,
                        names[0],
                        dst + target.removesuffix(".weight"),
                        projection,
                        src,
                    )
                )
                continue
            bindings.append(
                RuntimeTensorBinding(
                    names,
                    dst + target,
                    _identity,
                    lambda names=names, projection=projection, width=local_width, hidden=hidden: ffn_local(
                        names, projection, width, hidden
                    ),
                )
            )
    checkpoint.load_parameters(model, tuple(bindings), validators)
    if head is not None:
        head._explicit_checkpoint_loaded = True
