import logging
from typing import Any, Dict, Iterator, Tuple

import torch
import torch.nn as nn

from rtp_llm.model_loader.model_weight_info import ModelWeights
from rtp_llm.model_loader.weight_module import CustomAtomicWeight
from rtp_llm.models_py.model_desc.bert import BertModel
from rtp_llm.ops import ParallelismConfig
from rtp_llm.ops.compute_ops import PyModelInputs, PyModelOutputs
from rtp_llm.utils.model_weight import W

logger = logging.getLogger(__name__)


def _as_iter(weights: Any) -> Iterator[Tuple[str, torch.Tensor]]:
    if isinstance(weights, dict):
        return iter(weights.items())
    return iter(weights)


def _strip_known_prefix(name: str, model_prefix: str) -> str:
    for prefix in (model_prefix + ".", "bert.", "roberta."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _float_to_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.is_floating_point():
        return tensor.to(dtype)
    return tensor


class _BertNewLoaderBase(nn.Module):
    """New-loader wrapper for legacy BERT/Roberta PyModel.

    The new loader creates this object first, then calls load_weights(weights_iter).
    load_weights builds the ModelWeights object from the incoming checkpoint stream
    and creates the internal BertModel only after all required tensors are mapped.
    """

    model_prefix = "bert"

    def __init__(self, model_config, load_config):
        super().__init__()
        self.config = model_config
        self.load_config = load_config
        self.compute_dtype = getattr(load_config, "compute_dtype", torch.float16)
        self.parallelism_config = getattr(load_config, "parallelism_config", None)
        if self.parallelism_config is None:
            self.parallelism_config = ParallelismConfig()
            self.parallelism_config.tp_size = getattr(load_config, "tp_size", 1)
            self.parallelism_config.tp_rank = getattr(load_config, "tp_rank", 0)
            self.parallelism_config.ep_size = getattr(load_config, "ep_size", 1)
            self.parallelism_config.ep_rank = getattr(load_config, "ep_rank", 0)
            self.parallelism_config.world_size = max(self.parallelism_config.tp_size, 1)
            self.parallelism_config.local_world_size = self.parallelism_config.world_size
        self.model = None
        self.weights = None

    def _lookup(self, state: Dict[str, torch.Tensor], name: str) -> torch.Tensor:
        if name in state:
            return state[name]
        prefixed = self.model_prefix + "." + name
        if prefixed in state:
            return state[prefixed]
        for prefix in ("bert.", "roberta."):
            alt = prefix + name
            if alt in state:
                return state[alt]
        raise KeyError(name)

    def _optional_lookup(self, state: Dict[str, torch.Tensor], name: str):
        try:
            return self._lookup(state, name)
        except KeyError:
            return None

    def _create_model_weights(self, state: Dict[str, torch.Tensor]) -> ModelWeights:
        num_layers = int(getattr(self.config, "num_layers"))
        weights = ModelWeights(num_layers, "cpu", self.compute_dtype)

        def put_global(w_name: str, ckpt_name: str, *, optional: bool = False):
            tensor = self._optional_lookup(state, ckpt_name) if optional else self._lookup(state, ckpt_name)
            if tensor is not None:
                weights.set_global_weight(w_name, _float_to_dtype(tensor.contiguous(), self.compute_dtype))

        put_global(W.embedding, "embeddings.word_embeddings.weight")
        put_global(W.positional_embedding, "embeddings.position_embeddings.weight")
        put_global(W.token_type_embedding, "embeddings.token_type_embeddings.weight", optional=True)
        if W.token_type_embedding not in weights.global_weights:
            type_vocab_size = int(getattr(self.config, "type_vocab_size", 0) or 0)
            if type_vocab_size > 0:
                hidden_size = int(getattr(self.config, "hidden_size"))
                weights.set_global_weight(
                    W.token_type_embedding,
                    torch.zeros(type_vocab_size, hidden_size, dtype=self.compute_dtype),
                )
        put_global(W.pre_decoder_ln_gamma, "embeddings.LayerNorm.weight")
        put_global(W.pre_decoder_ln_beta, "embeddings.LayerNorm.bias")

        for i in range(num_layers):
            base = f"encoder.layer.{i}"
            q_w = self._lookup(state, f"{base}.attention.self.query.weight")
            k_w = self._lookup(state, f"{base}.attention.self.key.weight")
            v_w = self._lookup(state, f"{base}.attention.self.value.weight")
            q_b = self._lookup(state, f"{base}.attention.self.query.bias")
            k_b = self._lookup(state, f"{base}.attention.self.key.bias")
            v_b = self._lookup(state, f"{base}.attention.self.value.bias")

            weights.set_layer_weight(
                i,
                W.attn_qkv_w,
                torch.cat([q_w.t(), k_w.t(), v_w.t()], dim=1)
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_qkv_b,
                torch.cat([q_b, k_b, v_b], dim=0).contiguous().to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_o_w,
                self._lookup(state, f"{base}.attention.output.dense.weight")
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.attn_o_b,
                self._lookup(state, f"{base}.attention.output.dense.bias")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ln_gamma,
                self._lookup(state, f"{base}.attention.output.LayerNorm.weight")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ln_beta,
                self._lookup(state, f"{base}.attention.output.LayerNorm.bias")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_w3,
                self._lookup(state, f"{base}.intermediate.dense.weight")
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_b3,
                self._lookup(state, f"{base}.intermediate.dense.bias")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_w2,
                self._lookup(state, f"{base}.output.dense.weight")
                .t()
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.ffn_b2,
                self._lookup(state, f"{base}.output.dense.bias")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ffn_ln_gamma,
                self._lookup(state, f"{base}.output.LayerNorm.weight")
                .contiguous()
                .to(self.compute_dtype),
            )
            weights.set_layer_weight(
                i,
                W.post_ffn_ln_beta,
                self._lookup(state, f"{base}.output.LayerNorm.bias")
                .contiguous()
                .to(self.compute_dtype),
            )

        return weights

    def _add_custom_weights(self, model_weights: ModelWeights, custom_state: Dict[str, torch.Tensor]):
        for name, tensor in custom_state.items():
            if not isinstance(tensor, torch.Tensor):
                continue
            model_weights.set_global_weight(
                CustomAtomicWeight.prefix + name,
                _float_to_dtype(tensor.contiguous(), self.compute_dtype),
            )

    def _build_inner_model(self):
        self.model = BertModel(
            self.config,
            self.parallelism_config,
            self.weights,
            max_generate_batch_size=int(getattr(self.config, "max_generate_batch_size", 0) or 0),
            quant_config=getattr(self.config, "quant_config", None),
            fmha_config=getattr(self.load_config, "fmha_config", None),
            py_hw_kernel_config=getattr(self.load_config, "hw_kernel_config", None),
            device_resource_config=getattr(self.load_config, "device_resource_config", None),
        )

    def load_weights(self, weights):
        state: Dict[str, torch.Tensor] = {}
        custom_state: Dict[str, torch.Tensor] = {}
        dropped = 0
        for name, tensor in _as_iter(weights):
            stripped = _strip_known_prefix(name, self.model_prefix)
            if (
                stripped.startswith("embeddings.")
                or stripped.startswith("encoder.layer.")
            ):
                state[stripped] = tensor
            elif isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                custom_name = name if name.startswith("bert.pooler.") else stripped
                custom_state[custom_name] = tensor
            else:
                dropped += 1
        self.weights = self._create_model_weights(state)
        self._add_custom_weights(self.weights, custom_state)
        self._build_inner_model()
        logger.info(
            "%s newloader loaded BERT-style weights: tensors=%d custom_tensors=%d dropped=%d",
            self.__class__.__name__,
            len(state),
            len(custom_state),
            dropped,
        )

    def _move_model_weights(self, fn):
        if self.weights is None:
            return
        for k, v in list(self.weights.global_weights.items()):
            self.weights.global_weights[k] = fn(v)
        for layer in self.weights.weights:
            for k, v in list(layer.items()):
                layer[k] = fn(v)

    def _apply(self, fn):
        super()._apply(fn)
        if self.model is not None:
            self._move_model_weights(fn)
            self._build_inner_model()
        return self

    def initialize(self, init_resource):
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        return self.model.initialize(init_resource)

    def fill_params(self, *args, **kwargs):
        return self.model.fill_params(*args, **kwargs)

    def prepare_fmha_impl(self, *args, **kwargs):
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        return self.model.prepare_fmha_impl(*args, **kwargs)

    @staticmethod
    def _is_missing_tensor(tensor) -> bool:
        return tensor is None or (hasattr(tensor, "numel") and tensor.numel() == 0)

    def _fill_bert_embedding_inputs(self, inputs: PyModelInputs):
        bert_inputs = inputs.bert_embedding_inputs
        if self._is_missing_tensor(bert_inputs.position_encoding):
            bert_inputs.position_encoding = self.weights.get_global_weight(W.positional_embedding)
        if self._is_missing_tensor(bert_inputs.token_type_embedding):
            bert_inputs.token_type_embedding = self.weights.get_global_weight(W.token_type_embedding)
        return bert_inputs

    def forward(self, inputs: PyModelInputs, fmha_impl=None) -> PyModelOutputs:
        if self.model is None:
            raise RuntimeError("BERT newloader model is not loaded")
        self._fill_bert_embedding_inputs(inputs)
        return self.model(inputs, fmha_impl)


class BertForEmbedding(_BertNewLoaderBase):
    model_prefix = "bert"


class RobertaForEmbedding(_BertNewLoaderBase):
    model_prefix = "roberta"
