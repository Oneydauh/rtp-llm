import inspect
import logging
import os
import platform
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterator, List, Optional, Tuple

import rtp_llm.models_py.weight_mapper as weight_mapper
import torch
import torch.nn as nn
from rtp_llm.models_py.module_base import RtpModule, collect_loaded_tensor_ids
from rtp_llm.models_py.registry import get_model_class, list_models

logger = logging.getLogger(__name__)
_CUSTOM_WEIGHT_PREFIX = "__custom__."

_EXPERT_ID_RE = re.compile(r"(?:^|\.)experts\.(\d+)(?:\.|$)")
_STACKED_EXPERT_RE = re.compile(
    r"^(?P<prefix>(?:.*\.)?experts)\."
    r"(?P<projection>gate_up_proj|down_proj)"
    r"(?P<suffix>(?:\..+)?)$"
)
_FASTSAFETENSORS_REQUIRED_ARGUMENTS = frozenset(
    {
        "pg",
        "hf_weights_files",
        "use_tqdm_on_load",
        "device",
        "bbuf_size_kb",
        "use_shm",
        "nogds",
    }
)
_FASTSAFETENSORS_MIN_BBUF_SIZE_KB = 16 * 1024
_FASTSAFETENSORS_MAX_BBUF_SIZE_KB = 2 * 1024 * 1024


def _require_callable_name_filter(
    model_filter: Optional[Callable[[str], bool]], label: str
) -> Optional[Callable[[str], bool]]:
    if isinstance(model_filter, nn.Module):
        raise TypeError(f"{label} must be callable or None, not an nn.Module")
    if model_filter is not None and not callable(model_filter):
        raise TypeError(f"{label} must be callable or None")
    return model_filter


class _ExpertRangeFilter:
    """Select only this EP rank's per-expert checkpoint tensors."""

    def __init__(self, num_experts: int, ep_size: int, ep_rank: int):
        for name, value in (
            ("num_experts", num_experts),
            ("ep_size", ep_size),
            ("ep_rank", ep_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer, got {value!r}")
        if ep_size <= 0 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"Invalid EP partition: rank={ep_rank}, size={ep_size}")
        if num_experts <= 0 or num_experts % ep_size != 0:
            raise ValueError(
                f"num_experts={num_experts} must be positive and divisible by "
                f"ep_size={ep_size}"
            )
        experts_per_rank = num_experts // ep_size
        self.start_expert = ep_rank * experts_per_rank
        self.end_expert = self.start_expert + experts_per_rank
        self.ep_size = ep_size
        self.num_experts = num_experts

    def should_load(self, name: str) -> bool:
        match = _EXPERT_ID_RE.search(name)
        if match is not None:
            expert_id = int(match.group(1))
            if not 0 <= expert_id < self.num_experts:
                raise ValueError(
                    f"Checkpoint tensor {name!r} has expert id {expert_id} outside "
                    f"[0, {self.num_experts})"
                )
            return self.start_expert <= expert_id < self.end_expert
        if self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name):
            raise ValueError(
                "EP loading requires per-expert checkpoint keys; stacked all-expert "
                f"tensor {name!r} would materialize non-local experts"
            )
        return True

    def handles_safetensor(self, name: str) -> bool:
        return self.ep_size > 1 and _STACKED_EXPERT_RE.fullmatch(name) is not None

    def expand_safetensor(
        self, name: str, shape: Tuple[int, ...]
    ) -> List[Tuple[str, int]]:
        match = _STACKED_EXPERT_RE.fullmatch(name)
        if match is None:
            raise ValueError(f"Tensor {name!r} is not a stacked expert tensor")
        if not shape or shape[0] != self.num_experts:
            raise ValueError(
                f"Stacked expert tensor {name!r} must contain "
                f"{self.num_experts} experts, got shape {shape}"
            )
        suffix = match.group("suffix") or ".weight"
        prefix = match.group("prefix")
        projection = match.group("projection")
        return [
            (f"{prefix}.{expert_id}.{projection}{suffix}", expert_id)
            for expert_id in range(self.start_expert, self.end_expert)
        ]


def _validate_runtime_device(device: str, label: str) -> None:
    if not isinstance(device, str) or not device.strip():
        raise ValueError(f"{label} must be a non-empty string")
    try:
        parsed = torch.device(device)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} {device!r}") from exc
    if parsed.type == "meta":
        raise ValueError(
            f"{label} cannot be meta; newloader requires materialized weights"
        )


class NewLoaderLoadMethod(str, Enum):
    AUTO = "auto"
    SCRATCH = "scratch"
    FASTSAFETENSORS = "fastsafetensors"


@dataclass(frozen=True)
class NewLoaderConfig:
    tp_size: int = 1
    tp_rank: int = 0
    ep_size: int = 1
    ep_rank: int = 0
    compute_dtype: torch.dtype = torch.float16
    device: str = "cuda"
    load_method: NewLoaderLoadMethod = NewLoaderLoadMethod.AUTO
    quant_config: Any = None
    parallelism_config: Any = None
    moe_config: Any = None
    fmha_config: Any = None
    device_resource_config: Any = None
    attn_tp_size: Optional[int] = None
    attn_tp_rank: Optional[int] = None
    ffn_tp_size: Optional[int] = None
    ffn_tp_rank: Optional[int] = None
    lm_head_tp_size: Optional[int] = None
    lm_head_tp_rank: Optional[int] = None
    keep_mla_checkpoint_weights: bool = False
    custom_weight_mappings: Tuple[Tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.load_method, str):
            try:
                object.__setattr__(
                    self,
                    "load_method",
                    NewLoaderLoadMethod(self.load_method.strip().lower()),
                )
            except ValueError as exc:
                raise ValueError(
                    f"Unsupported newloader load method {self.load_method!r}"
                ) from exc
        elif not isinstance(self.load_method, NewLoaderLoadMethod):
            raise TypeError(
                f"load_method must be NewLoaderLoadMethod or str, got "
                f"{type(self.load_method).__name__}"
            )
        for name in ("tp_size", "tp_rank", "ep_size", "ep_rank"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(
                    f"{name} must be an integer, got {type(value).__name__}"
                )
        if self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError(
                f"Invalid TP partition: rank={self.tp_rank}, size={self.tp_size}"
            )
        for prefix in ("attn_tp", "ffn_tp", "lm_head_tp"):
            size_name = f"{prefix}_size"
            rank_name = f"{prefix}_rank"
            size = getattr(self, size_name)
            rank = getattr(self, rank_name)
            if size is None and rank is None:
                size, rank = self.tp_size, self.tp_rank
                object.__setattr__(self, size_name, size)
                object.__setattr__(self, rank_name, rank)
            elif size is None or rank is None:
                raise ValueError(
                    f"{size_name} and {rank_name} must be configured together"
                )
            if isinstance(size, bool) or not isinstance(size, int):
                raise TypeError(f"{size_name} must be an integer")
            if isinstance(rank, bool) or not isinstance(rank, int):
                raise TypeError(f"{rank_name} must be an integer")
            if size <= 0 or not 0 <= rank < size:
                raise ValueError(
                    f"Invalid {prefix} partition: rank={rank}, size={size}"
                )
            if prefix in ("attn_tp", "ffn_tp"):
                if size not in (1, self.tp_size):
                    raise ValueError(
                        f"{size_name}={size} must be either 1 or the physical "
                        f"tp_size={self.tp_size}; independent TP subgroups are "
                        "not supported by Group.TP collectives"
                    )
                expected_rank = 0 if size == 1 else self.tp_rank
                if rank != expected_rank:
                    raise ValueError(
                        f"{rank_name}={rank} does not match the supported "
                        f"{prefix} topology: expected rank={expected_rank} for "
                        f"size={size}"
                    )
        if self.ep_size <= 0 or not 0 <= self.ep_rank < self.ep_size:
            raise ValueError(
                f"Invalid EP partition: rank={self.ep_rank}, size={self.ep_size}"
            )
        if not isinstance(self.compute_dtype, torch.dtype):
            raise TypeError("compute_dtype must be a torch.dtype")
        if not isinstance(self.keep_mla_checkpoint_weights, bool):
            raise TypeError("keep_mla_checkpoint_weights must be a bool")
        if not isinstance(self.custom_weight_mappings, tuple):
            raise TypeError("custom_weight_mappings must be a tuple")
        runtime_names = set()
        checkpoint_names = set()
        for mapping in self.custom_weight_mappings:
            if not isinstance(mapping, tuple) or len(mapping) != 2:
                raise TypeError(
                    "custom_weight_mappings entries must be "
                    "(runtime_name, checkpoint_name) tuples"
                )
            runtime_name, checkpoint_name = mapping
            if not isinstance(runtime_name, str) or not runtime_name:
                raise TypeError("custom runtime weight names must be non-empty strings")
            if not runtime_name.startswith(_CUSTOM_WEIGHT_PREFIX):
                raise ValueError(
                    f"custom runtime weight {runtime_name!r} must start with "
                    f"{_CUSTOM_WEIGHT_PREFIX!r}"
                )
            if not isinstance(checkpoint_name, str) or not checkpoint_name:
                raise TypeError(
                    "custom checkpoint weight names must be non-empty strings"
                )
            if runtime_name in runtime_names:
                raise ValueError(f"Duplicate custom runtime weight {runtime_name!r}")
            if checkpoint_name in checkpoint_names:
                raise ValueError(
                    f"Duplicate custom checkpoint weight {checkpoint_name!r}"
                )
            runtime_names.add(runtime_name)
            checkpoint_names.add(checkpoint_name)
        _validate_runtime_device(self.device, "device")
        if self.parallelism_config is not None:
            for prefix in ("tp", "ep"):
                configured_size = getattr(
                    self.parallelism_config, f"{prefix}_size", None
                )
                configured_rank = getattr(
                    self.parallelism_config, f"{prefix}_rank", None
                )
                if configured_size is None and configured_rank is None:
                    continue
                expected = (
                    (self.tp_size, self.tp_rank)
                    if prefix == "tp"
                    else (self.ep_size, self.ep_rank)
                )
                if (configured_size, configured_rank) != expected:
                    raise ValueError(
                        f"parallelism_config {prefix.upper()} partition does not "
                        f"match NewLoaderConfig: parallelism="
                        f"({configured_rank}, {configured_size}) loader="
                        f"({expected[1]}, {expected[0]})"
                    )
            topology_getters = {
                "attn_tp": ("get_attn_tp_size", "get_attn_tp_rank"),
                "ffn_tp": ("get_ffn_tp_size", "get_ffn_tp_rank"),
                "lm_head_tp": ("tp_size", "tp_rank"),
            }
            for prefix, (
                size_getter_name,
                rank_getter_name,
            ) in topology_getters.items():
                size_source = getattr(self.parallelism_config, size_getter_name, None)
                rank_source = getattr(self.parallelism_config, rank_getter_name, None)
                if callable(size_source) and callable(rank_source):
                    actual = (size_source(), rank_source())
                elif size_source is not None and rank_source is not None:
                    actual = (size_source, rank_source)
                else:
                    continue
                expected = (
                    getattr(self, f"{prefix}_size"),
                    getattr(self, f"{prefix}_rank"),
                )
                if actual != expected:
                    raise ValueError(
                        f"parallelism_config {prefix} partition does not match "
                        f"NewLoaderConfig: parallelism=({actual[1]}, {actual[0]}) "
                        f"loader=({expected[1]}, {expected[0]})"
                    )


class NewModelLoader:
    """Create a PyModel first, stream tensors into it, then run post-load hooks."""

    def __init__(
        self,
        model_config: Any,
        load_config: Optional[NewLoaderConfig] = None,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        effective_config = NewLoaderConfig() if load_config is None else load_config
        if not isinstance(effective_config, NewLoaderConfig):
            raise TypeError(
                f"load_config must be NewLoaderConfig, got {type(effective_config).__name__}"
            )
        if device is not None:
            _validate_runtime_device(device, "device override")
            effective_config = replace(effective_config, device=device)
        self.model_config = model_config
        self.load_config = effective_config
        self.model_path = model_path
        self._ckpt_files = None

    def _resolve_load_method(self) -> NewLoaderLoadMethod:
        configured = self.load_config.load_method
        if configured == NewLoaderLoadMethod.AUTO:
            env_method = os.environ.get("LOAD_METHOD", "").strip().lower()
            if env_method:
                try:
                    configured = NewLoaderLoadMethod(env_method)
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported LOAD_METHOD environment value {env_method!r}"
                    ) from exc
        return configured

    def _model_type(self) -> str:
        value = (
            self.model_config.get("model_type", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_type", "")
        )
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model_config.model_type must be a non-empty string")
        return value.strip()

    def _resolved_model_path(self) -> str:
        if self.model_path:
            return self.model_path
        value = (
            self.model_config.get("model_path", "")
            if isinstance(self.model_config, dict)
            else getattr(self.model_config, "model_path", "")
        )
        if not value:
            raise ValueError("model_path is required")
        return value

    def _checkpoint_files(self):
        if self._ckpt_files is None:
            self._ckpt_files = weight_mapper.discover_ckpt_files(
                self._resolved_model_path(),
                tp_rank=self.load_config.tp_rank,
                tp_size=self.load_config.tp_size,
            )
            if not self._ckpt_files:
                raise FileNotFoundError(
                    f"No supported checkpoint files in {self._resolved_model_path()}"
                )
        return self._ckpt_files

    def _expert_filter(self):
        if self.load_config.ep_size == 1:
            return None
        num_experts = (
            self.model_config.get("num_experts", 0)
            if isinstance(self.model_config, dict)
            else getattr(
                self.model_config,
                "expert_num",
                getattr(self.model_config, "num_experts", 0),
            )
        )
        if isinstance(num_experts, bool) or not isinstance(num_experts, int):
            raise TypeError(
                f"model_config num_experts must be an integer, got {num_experts!r}"
            )
        if num_experts < 0:
            raise ValueError(
                f"model_config num_experts must be non-negative, got {num_experts}"
            )
        if num_experts == 0:
            model_type = (
                self.model_config.get("model_type", "")
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "model_type", "")
            )
            moe_top_k = (
                self.model_config.get(
                    "moe_k", self.model_config.get("num_experts_per_tok", 0)
                )
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_k", 0)
            )
            moe_style = (
                self.model_config.get("moe_style", 0)
                if isinstance(self.model_config, dict)
                else getattr(self.model_config, "moe_style", 0)
            )
            for name, value in (("moe_k", moe_top_k), ("moe_style", moe_style)):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(
                        f"model_config {name} must be an integer, got {value!r}"
                    )
                if value < 0:
                    raise ValueError(
                        f"model_config {name} must be non-negative, got {value}"
                    )
            if "moe" in str(model_type).lower() or moe_top_k > 0 or moe_style > 0:
                raise ValueError(
                    "EP is configured for a MoE model "
                    f"(model_type={model_type!r}, moe_k={moe_top_k}, "
                    f"moe_style={moe_style}), but "
                    "model_config reports zero experts"
                )
            logger.warning(
                "EP size %d is configured, but model_config reports zero experts; "
                "model_type=%r and zero moe_k/moe_style markers identify a dense "
                "model, so expert checkpoint filtering is disabled",
                self.load_config.ep_size,
                model_type,
            )
            return None
        return _ExpertRangeFilter(
            num_experts,
            self.load_config.ep_size,
            self.load_config.ep_rank,
        )

    @staticmethod
    def _model_checkpoint_name_filter(
        model: RtpModule,
    ) -> Optional[Callable[[str], bool]]:
        return _require_callable_name_filter(
            model.checkpoint_weight_name_filter(),
            f"{type(model).__name__}.checkpoint_weight_name_filter() return value",
        )

    @staticmethod
    def _checkpoint_name_filter(
        model_filter: Optional[Callable[[str], bool]],
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> Optional[Callable[[str], bool]]:
        model_filter = _require_callable_name_filter(model_filter, "model_filter")
        if model_filter is None and expert_filter is None:
            return None

        def should_load(name: str) -> bool:
            if model_filter is not None and not model_filter(name):
                return False
            if expert_filter is None:
                return True
            if expert_filter.handles_safetensor(name):
                return True
            return expert_filter.should_load(name)

        return should_load

    def _validate_ep_checkpoint_format(
        self,
        checkpoint_files,
        expert_filter: Optional[_ExpertRangeFilter],
    ) -> None:
        if expert_filter is None:
            return
        unsupported = [
            path
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if unsupported:
            raise ValueError(
                "EP streaming requires safetensors checkpoints with per-expert "
                "keys; PyTorch checkpoints are deserialized as a whole before "
                f"name filtering and would materialize non-local experts: {unsupported}"
            )

    def _create_model(self) -> nn.Module:
        model_type = self._model_type()
        try:
            model_cls = get_model_class(model_type)
        except KeyError as exc:
            raise ValueError(
                f"Model type {model_type!r} is not registered; available={list_models()}"
            ) from exc
        model = model_cls(self.model_config, self.load_config)
        if not isinstance(model, RtpModule):
            raise TypeError(
                f"Registered model {model_cls.__name__} must inherit RtpModule to "
                "provide newloader completeness validation"
            )
        if self.load_config.custom_weight_mappings and not bool(
            getattr(model, "supports_custom_weight_mappings", False)
        ):
            raise NotImplementedError(
                f"{type(model).__name__} does not support downstream custom weights"
            )
        try:
            source = inspect.getfile(model_cls)
        except (TypeError, OSError):
            source = "<unknown>"
        logger.info(
            "Created newloader model %s from %s", model_cls.__qualname__, source
        )
        return model

    @staticmethod
    def _validate_loaded_weights(model: nn.Module) -> None:
        loaded_tensor_ids = frozenset(collect_loaded_tensor_ids(model))
        root_validator = getattr(model, "validate_weights_loaded", None)
        if not callable(root_validator):
            raise TypeError(
                f"Registered model {type(model).__name__} has no weight completeness validator"
            )
        root_validator(loaded_tensor_ids)

        for module_name, module in model.named_modules():
            if module is model:
                continue
            custom_loader = getattr(type(module), "load_weights", None)
            if (
                custom_loader is not None
                and custom_loader is not RtpModule.load_weights
            ):
                validator = getattr(module, "validate_weights_loaded", None)
                if not callable(validator):
                    raise TypeError(
                        f"Custom weight loader {type(module).__name__}.load_weights() "
                        "must define validate_weights_loaded()"
                    )
                try:
                    validator(loaded_tensor_ids)
                except Exception as exc:
                    context = (
                        f"Weight validation failed for {module_name} "
                        f"({type(module).__name__})"
                    )
                    if exc.args:
                        exc.args = (f"{context}: {exc.args[0]}", *exc.args[1:])
                    else:
                        exc.args = (context,)
                    raise

    @staticmethod
    def _run_post_load_hooks(model: nn.Module) -> None:
        for module in model.modules():
            hook = getattr(module, "process_weights_after_loading", None)
            if callable(hook):
                hook()

    @staticmethod
    def _requires_staged_device_postprocess(model: nn.Module) -> bool:
        return any(
            isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
            for module in model.modules()
        )

    @staticmethod
    def _fastsafetensors_api():
        try:
            from fastsafetensors import ParallelLoader, SingleGroup
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            raise RuntimeError(
                "the fastsafetensors package is unavailable or failed to import: "
                f"{exc}"
            ) from exc

        for symbol_name, symbol in (
            ("ParallelLoader", ParallelLoader),
            ("SingleGroup", SingleGroup),
        ):
            if not callable(symbol):
                raise RuntimeError(f"fastsafetensors.{symbol_name} must be callable")
        try:
            parameters = inspect.signature(ParallelLoader.__init__).parameters
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "cannot inspect fastsafetensors.ParallelLoader constructor"
            ) from exc
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        missing = _FASTSAFETENSORS_REQUIRED_ARGUMENTS - set(parameters)
        if missing and not accepts_kwargs:
            raise RuntimeError(
                "fastsafetensors.ParallelLoader is incompatible with NewLoader; "
                f"missing constructor arguments {sorted(missing)}"
            )
        return ParallelLoader, SingleGroup

    @staticmethod
    def _safetensor_metadata_summary(
        model_path: str, checkpoint_files: List[str]
    ) -> Tuple[List[str], List[str]]:
        """Return duplicate names and stacked expert names without reading data.

        A valid Hugging Face index is the preferred metadata source.  The
        streaming iterator still detects duplicate names while loading, so an
        index avoids a second full header walk without weakening correctness.
        Header inspection remains the fallback for unindexed checkpoints.
        """
        selected_identities = {
            os.path.realpath(os.fspath(path)) for path in checkpoint_files
        }
        index_entries = weight_mapper.safetensor_index_entries(model_path)
        if index_entries is not None:
            indexed_identities = {
                os.path.realpath(shard_path) for _, shard_path in index_entries
            }
            if selected_identities.issubset(indexed_identities):
                stacked = [
                    name
                    for name, shard_path in index_entries
                    if os.path.realpath(shard_path) in selected_identities
                    and _STACKED_EXPERT_RE.fullmatch(name)
                ]
                return [], stacked

        from safetensors import safe_open

        seen = set()
        duplicates = set()
        stacked = []
        for path in checkpoint_files:
            with safe_open(path, framework="pt", device="cpu") as handle:
                for name in handle.keys():
                    if name in seen:
                        duplicates.add(name)
                    seen.add(name)
                    if _STACKED_EXPERT_RE.fullmatch(name):
                        stacked.append(name)
        return sorted(duplicates), sorted(stacked)

    def _fastsafetensors_distributed_reason(self) -> Optional[str]:
        """Validate that WORLD is exactly the loader's TP/EP participant set."""
        expected_world_size = max(
            self.load_config.tp_size,
            self.load_config.ep_size,
        )
        if not torch.distributed.is_initialized():
            if expected_world_size > 1:
                return (
                    "TP/EP fastsafetensors loading requires an initialized "
                    f"distributed WORLD group of size {expected_world_size}"
                )
            return None

        world_size = torch.distributed.get_world_size()
        if world_size != expected_world_size:
            return (
                "distributed WORLD group does not match the NewLoader TP/EP "
                f"topology: world_size={world_size}, expected={expected_world_size}"
            )
        world_rank = torch.distributed.get_rank()
        expected_tp_rank = world_rank % self.load_config.tp_size
        expected_ep_rank = world_rank % self.load_config.ep_size
        if (
            self.load_config.tp_rank != expected_tp_rank
            or self.load_config.ep_rank != expected_ep_rank
        ):
            return (
                "distributed WORLD rank does not match the NewLoader TP/EP "
                f"ranks: world_rank={world_rank}, loader_tp_rank="
                f"{self.load_config.tp_rank}, expected_tp_rank={expected_tp_rank}, "
                f"loader_ep_rank={self.load_config.ep_rank}, "
                f"expected_ep_rank={expected_ep_rank}"
            )
        return None

    @staticmethod
    def _fastsafetensors_bbuf_size_kb(checkpoint_files: List[str]) -> int:
        max_shard_bytes = max(os.path.getsize(path) for path in checkpoint_files)
        max_shard_kb = (max_shard_bytes + 1023) // 1024
        return min(
            _FASTSAFETENSORS_MAX_BBUF_SIZE_KB,
            max(_FASTSAFETENSORS_MIN_BBUF_SIZE_KB, max_shard_kb),
        )

    @staticmethod
    def _cuda_cufile_available() -> bool:
        """Use fastsafetensors' native probe as the GDS source of truth."""
        try:
            from fastsafetensors import cpp as fastsafetensors_cpp
        except (ImportError, OSError, RuntimeError):
            return False
        try:
            return bool(fastsafetensors_cpp.is_cufile_found())
        except (AttributeError, OSError, RuntimeError):
            return False

    @staticmethod
    def _accelerator_backend_name() -> str:
        """Identify CUDA-compatible runtimes without importing the device package."""
        if torch.version.hip is not None:
            return "ROCm"
        if (
            os.environ.get("PPU_HOME")
            or "ppu" in getattr(torch, "__version__", "").lower()
        ):
            return "Ppu"
        return "Cuda"

    @staticmethod
    def _fastsafetensors_use_nogds(device: torch.device) -> bool:
        if device.type == "cpu":
            return True
        if os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1":
            return True
        if (
            device.type == "cuda"
            and torch.version.hip is None
            and not NewModelLoader._cuda_cufile_available()
        ):
            logger.warning(
                "libcufile.so is unavailable; NewLoader fastsafetensors is "
                "falling back to the CUDA nogds path"
            )
            return True
        return False

    @staticmethod
    def _model_storage_bytes(model: nn.Module) -> int:
        seen = set()
        total = 0
        tensors = list(
            model.named_parameters(recurse=True, remove_duplicate=False)
        ) + list(model.named_buffers(recurse=True, remove_duplicate=False))
        for _, tensor in tensors:
            if tensor is None or id(tensor) in seen:
                continue
            if tensor.is_meta:
                raise RuntimeError(
                    "model contains meta tensors that cannot be sized for automatic "
                    "fastsafetensors selection"
                )
            seen.add(id(tensor))
            total += tensor.numel() * tensor.element_size()
        return total

    def _fastsafetensors_prerequisite_reason(
        self,
        model: nn.Module,
        checkpoint_files: List[str],
        *,
        automatic: bool,
        expert_filter: Optional[_ExpertRangeFilter] = None,
    ) -> Optional[str]:
        if not checkpoint_files:
            return "checkpoint shard selection is empty"
        non_safetensors = [
            os.fspath(path)
            for path in checkpoint_files
            if not os.fspath(path).lower().endswith(".safetensors")
        ]
        if non_safetensors:
            return f"checkpoint contains non-safetensors files {non_safetensors}"
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files):
            return (
                "rank-local consolidated checkpoints cannot use the collective "
                "WORLD fastsafetensors loader; use scratch until a rank-local "
                "SingleGroup path is validated"
            )
        if self._requires_staged_device_postprocess(model):
            return (
                "the model requires staged device postprocessing (for example "
                "online quantization), which is currently memory-bounded only by "
                "the scratch path"
            )

        target_device = torch.device(self.load_config.device)
        if target_device.type not in ("cpu", "cuda"):
            return f"runtime device {target_device} is not supported"
        if automatic and target_device.type == "cpu":
            return (
                "the runtime device is CPU, where direct accelerator loading "
                "has no benefit"
            )
        if target_device.type == "cuda":
            if not torch.cuda.is_available():
                return f"runtime device {target_device} is unavailable"
            backend = self._accelerator_backend_name()
            if automatic and backend != "Cuda":
                return (
                    "automatic fastsafetensors selection is disabled on the "
                    f"unvalidated {backend} backend; request "
                    "fastsafetensors explicitly to override"
                )
            machine = platform.machine().lower()
            if automatic and machine not in ("amd64", "x86_64"):
                return (
                    "automatic fastsafetensors selection is disabled on the "
                    f"unvalidated {machine or 'unknown'} architecture; request "
                    "fastsafetensors explicitly to override"
                )
            if (
                torch.version.hip is not None
                and os.environ.get("FASTSAFETENSORS_NOGDS", "0") == "1"
            ):
                return (
                    "FASTSAFETENSORS_NOGDS=1 is unsupported on ROCm by the "
                    "installed fastsafetensors backend"
                )
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            device_count = torch.cuda.device_count()
            if not 0 <= device_index < device_count:
                return (
                    f"runtime device index {device_index} is outside the available "
                    f"CUDA device range [0, {device_count})"
                )

        distributed_reason = self._fastsafetensors_distributed_reason()
        if distributed_reason is not None:
            return distributed_reason

        try:
            self._fastsafetensors_api()
        except RuntimeError as exc:
            return str(exc)
        try:
            duplicates, stacked_experts = self._safetensor_metadata_summary(
                self._resolved_model_path(), checkpoint_files
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return f"failed to inspect safetensors metadata: {exc}"
        if duplicates:
            sample = duplicates[:10]
            suffix = (
                f" (+{len(duplicates) - len(sample)} more)"
                if len(duplicates) > 10
                else ""
            )
            return f"checkpoint tensor names are duplicated: {sample}{suffix}"
        if expert_filter is not None and stacked_experts:
            sample = stacked_experts[:10]
            suffix = (
                f" (+{len(stacked_experts) - len(sample)} more)"
                if len(stacked_experts) > len(sample)
                else ""
            )
            return (
                "EP fastsafetensors loading does not support stacked all-expert "
                "tensors because ParallelLoader materializes non-local experts "
                f"before slicing: {sample}{suffix}; use scratch or a per-expert "
                "checkpoint"
            )
        if automatic:
            device_index = (
                torch.cuda.current_device()
                if target_device.index is None
                else target_device.index
            )
            try:
                free_bytes, _ = torch.cuda.mem_get_info(device_index)
                model_bytes = self._model_storage_bytes(model)
                max_shard_bytes = max(
                    os.path.getsize(path) for path in checkpoint_files
                )
                bbuf_bytes = self._fastsafetensors_bbuf_size_kb(checkpoint_files) * 1024
            except (OSError, RuntimeError, ValueError) as exc:
                return f"cannot determine fastsafetensors memory safety: {exc}"
            required_bytes = model_bytes + 3 * max(max_shard_bytes, bbuf_bytes)
            logger.info(
                "NewLoader fastsafetensors memory check: free=%.2fGiB, "
                "model=%.2fGiB, max_shard=%.2fGiB, bbuf=%.2fGiB, "
                "required=%.2fGiB",
                free_bytes / (1024**3),
                model_bytes / (1024**3),
                max_shard_bytes / (1024**3),
                bbuf_bytes / (1024**3),
                required_bytes / (1024**3),
            )
            if free_bytes <= required_bytes:
                return (
                    f"insufficient accelerator memory: free={free_bytes} bytes, "
                    f"required>{required_bytes} bytes"
                )
        return None

    @staticmethod
    def _synchronize_fastsafetensors_prerequisite(
        local_reason: Optional[str],
    ) -> Optional[str]:
        """Require every distributed rank to make the same loader decision."""
        if not torch.distributed.is_initialized():
            return local_reason
        world_size = torch.distributed.get_world_size()
        if world_size <= 1:
            return local_reason
        reasons: List[Optional[str]] = [None] * world_size
        try:
            torch.distributed.all_gather_object(reasons, local_reason)
        except (RuntimeError, TypeError, ValueError) as exc:
            return (
                "cannot synchronize fastsafetensors prerequisites across ranks: "
                f"{exc}"
            )
        failures = [
            f"rank {rank}: {reason}"
            for rank, reason in enumerate(reasons)
            if reason is not None
        ]
        if not failures:
            return None
        sample = failures[:4]
        suffix = (
            f" (+{len(failures) - len(sample)} more)"
            if len(failures) > len(sample)
            else ""
        )
        return f"distributed prerequisite failure: {'; '.join(sample)}{suffix}"

    def _select_load_method(
        self,
        requested: NewLoaderLoadMethod,
        model: nn.Module,
        checkpoint_files: List[str],
        expert_filter: Optional[_ExpertRangeFilter] = None,
    ) -> NewLoaderLoadMethod:
        if requested == NewLoaderLoadMethod.SCRATCH:
            return requested
        automatic = requested == NewLoaderLoadMethod.AUTO
        reason = self._fastsafetensors_prerequisite_reason(
            model,
            checkpoint_files,
            automatic=automatic,
            expert_filter=expert_filter,
        )
        reason = self._synchronize_fastsafetensors_prerequisite(reason)
        if reason is None:
            selected = NewLoaderLoadMethod.FASTSAFETENSORS
            logger.info(
                "NewLoader selected fastsafetensors (%s)",
                "automatic" if automatic else "explicit",
            )
            return selected
        if automatic:
            logger.info(
                "NewLoader automatic fastsafetensors selection fell back to "
                "scratch: %s",
                reason,
            )
            return NewLoaderLoadMethod.SCRATCH
        raise RuntimeError(f"NewLoader fastsafetensors prerequisite failed: {reason}")

    @staticmethod
    def _fastsafetensors_weights_iterator(
        checkpoint_files: List[str],
        device: str,
        name_filter: Optional[Callable[[str], bool]],
        safetensor_slice_expander: Optional[_ExpertRangeFilter],
    ) -> Iterator[Tuple[str, torch.Tensor]]:
        """Stream safetensors directly to the model's runtime device.

        The yielded tensor is consumed synchronously by ``RtpModule.load_weights``
        before the iterator advances. Stacked EP tensors must have been rejected
        by metadata preflight before this iterator constructs ``ParallelLoader``.
        """
        ParallelLoader, SingleGroup = NewModelLoader._fastsafetensors_api()

        if torch.distributed.is_initialized():
            process_group = torch.distributed.group.WORLD
        else:
            process_group = SingleGroup()

        target_device = torch.device(device)
        loader_device = str(target_device)
        if target_device.type == "cuda" and target_device.index is None:
            loader_device = f"cuda:{torch.cuda.current_device()}"
        use_nogds = NewModelLoader._fastsafetensors_use_nogds(target_device)
        bbuf_size_kb = NewModelLoader._fastsafetensors_bbuf_size_kb(checkpoint_files)
        loader = ParallelLoader(
            pg=process_group,
            hf_weights_files=sorted(os.fspath(path) for path in checkpoint_files),
            use_tqdm_on_load=True,
            device=loader_device,
            bbuf_size_kb=bbuf_size_kb,
            use_shm=not use_nogds,
            nogds=use_nogds,
        )
        seen = set()
        try:
            for name, tensor in loader.iterate_weights():
                if name_filter is not None and not name_filter(name):
                    continue
                if (
                    safetensor_slice_expander is not None
                    and safetensor_slice_expander.handles_safetensor(name)
                ):
                    raise RuntimeError(
                        "EP fastsafetensors preflight failed to reject stacked "
                        f"all-expert tensor {name!r} before device materialization"
                    )
                if name in seen:
                    raise RuntimeError(
                        f"Checkpoint tensor {name!r} appears more than once during "
                        "fastsafetensors loading"
                    )
                seen.add(name)
                yield name, tensor
        finally:
            backing_loader = getattr(loader, "loader", None)
            close = getattr(backing_loader, "close", None)
            if callable(close):
                close()

    @staticmethod
    def _validate_runtime_backends(model: nn.Module, device: str) -> None:
        target_device = torch.device(device)
        for module in model.modules():
            if isinstance(module, RtpModule):
                module.validate_runtime_device(target_device)

    @staticmethod
    def _migrate_staged_modules(model: nn.Module, device: str) -> None:
        """Move and compress online-quantized leaves one at a time.

        Migrating the entire model first would retain every BF16 staging weight on
        the accelerator until post-load hooks run. Processing each marked leaf
        immediately bounds the extra accelerator memory to one staging layer.
        """
        target_device = torch.device(device)
        staged_modules = [
            module
            for module in model.modules()
            if isinstance(module, RtpModule)
            and module.requires_staged_device_postprocess()
        ]
        staged_set = set(staged_modules)
        tensor_registrations = {}
        for module in model.modules():
            tensors = list(
                module.named_parameters(recurse=False, remove_duplicate=False)
            ) + list(module.named_buffers(recurse=False, remove_duplicate=False))
            for name, tensor in tensors:
                if tensor is not None:
                    tensor_registrations.setdefault(id(tensor), []).append(
                        (module, name)
                    )
        for registrations in tensor_registrations.values():
            if (
                any(module in staged_set for module, _ in registrations)
                and len(registrations) > 1
            ):
                aliases = sorted(
                    f"{type(module).__name__}.{name}" for module, name in registrations
                )
                raise RuntimeError(
                    "Online-quantized staging does not support shared tensor "
                    f"aliases: {aliases}"
                )

        for module in staged_modules:
            if any(True for _ in module.children()):
                raise RuntimeError(
                    "Staged device postprocess is supported only for leaf modules, "
                    f"got {type(module).__name__}"
                )
            module.to(target_device)
            module.process_weights_after_loading()

    @torch.inference_mode()
    def load(self) -> nn.Module:
        requested_method = self._resolve_load_method()
        checkpoint_files = self._checkpoint_files()
        expert_filter = self._expert_filter()
        self._validate_ep_checkpoint_format(checkpoint_files, expert_filter)
        with torch.device("cpu"):
            model = self._create_model()
        if weight_mapper.is_rank_local_checkpoint(checkpoint_files) and not bool(
            getattr(model, "supports_rank_local_checkpoint", False)
        ):
            raise ValueError(
                f"{type(model).__name__} does not support rank-local consolidated "
                "checkpoints; use a global HF checkpoint"
            )
        self._validate_runtime_backends(model, self.load_config.device)
        model_filter = self._model_checkpoint_name_filter(model)
        name_filter = self._checkpoint_name_filter(model_filter, expert_filter)
        selected_files = weight_mapper.select_safetensor_files(
            self._resolved_model_path(), checkpoint_files, model_filter
        )
        if len(selected_files) != len(checkpoint_files):
            logger.info(
                "Model checkpoint filter selected %d/%d shards",
                len(selected_files),
                len(checkpoint_files),
            )
        method = self._select_load_method(
            requested_method,
            model,
            selected_files,
            expert_filter,
        )
        logger.info(
            "NewLoader load method: requested=%s, selected=%s",
            requested_method.value,
            method.value,
        )
        started = time.time()
        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            model.to(self.load_config.device)
        if method == NewLoaderLoadMethod.FASTSAFETENSORS:
            weights = self._fastsafetensors_weights_iterator(
                selected_files,
                self.load_config.device,
                name_filter,
                expert_filter,
            )
        else:
            weights = weight_mapper.get_all_weights(
                selected_files,
                device="cpu",
                name_filter=name_filter,
                safetensor_slice_expander=expert_filter,
            )
        try:
            model.load_weights(weights)
        finally:
            close_weights = getattr(weights, "close", None)
            if callable(close_weights):
                close_weights()
        self._validate_loaded_weights(model)
        logger.info(
            "Streamed checkpoint tensors with %s in %.2fs",
            method.value,
            time.time() - started,
        )

        if method == NewLoaderLoadMethod.SCRATCH:
            self._migrate_staged_modules(model, self.load_config.device)
        model.to(self.load_config.device)
        self._validate_loaded_weights(model)
        self._run_post_load_hooks(model)
        model.eval()
        return model
