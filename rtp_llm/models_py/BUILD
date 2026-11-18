load("@arch_config//:arch_select.bzl", "requirement", "triton_deps")
load("@local_config_cuda//cuda:build_defs.bzl", "cuda_default_copts_without_arch")

# Portable checkpoint protocol. Keep this target independent of the native
# inference library so real CPU checkpoint regressions need no fake ABI.
py_library(
    name = "new_loader_core",
    srcs = [
        "__init__.py",
        "borrowed_parameters.py",
        "downstream_heads.py",
        "expert_placement.py",
        "fastsafetensors_read_spec.py",
        "finetune_overlay.py",
        "fst_range_reader.py",
        "fst_file_plan.py",
        "ft_checkpoint.py",
        "lora_checkpoint.py",
        "meta_checkpoint.py",
        "model_loader.py",
        "module_base.py",
        "read_spec.py",
        "registry.py",
        "runtime_weight_update.py",
        "weight_mapper.py",
    ],
    deps = [
        "//rtp_llm:device_detection",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm/models_py/distributed:rank_layout",
    ] + select({
        "@//:using_ppu": [":ppu_entrypoint"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_test(
    name = "direct_lora_checkpoint_test",
    srcs = ["direct_lora_checkpoint_test.py"],
    deps = [":new_weight_loader", "//rtp_llm:lora", "//rtp_llm:utils"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
)

py_test(
    name = "main_core_checkpoint_test",
    srcs = ["main_core_checkpoint_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    deps = [":new_loader_core"],
)

py_test(
    name = "ft_module_install_test",
    srcs = ["ft_module_install_test.py"],
    deps = [":new_loader_core"],
)

py_test(
    name = "rank_local_ft_test",
    srcs = ["rank_local_ft_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        ":new_loader_core",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

py_test(
    name = "main_parallel_runtime_test",
    srcs = ["main_parallel_runtime_test.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    },
    deps = [
        ":new_loader_core",
        "//rtp_llm:fastsafetensors",
    ],
)

py_test(
    name = "qwen2_main_ft_stage_test",
    srcs = ["qwen2_main_ft_stage_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        ":new_weight_loader",
        "//rtp_llm/tools/convert:ft_export_metadata",
        "//rtp_llm/tools/convert:ft_export_writer",
    ],
)

# Direct parameter owners use main's existing runtime models/helpers, never a
# backported scheduler/cache/PP engine. Production routing is integrated later.
py_library(
    name = "new_weight_loader",
    srcs = ["eplb_controller.py", "new_weight_manager.py", "quantized_weight_update.py"] + glob([
        "layers/*.py",
        "new_models/*.py",
        "new_models/**/*.py",
        "quant_methods/*.py",
    ], exclude = [
        "**/test/**",
        "**/tests/**",
        "**/*_test.py",
        "**/test_*.py",
    ]),
    # Full model owners import both native binding modules. Place the DSOs
    # together in the package path used by rtp_llm.ops; the portable reader
    # above deliberately has no native inference dependency.
    data = [
        "//rtp_llm/libs:librtp_compute_ops_so",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    deps = [
        ":models",
        ":new_loader_core",
        "//rtp_llm:config",
        "//rtp_llm:eplb",
        "//rtp_llm/model_loader/tipc:tipc",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:ops",
        "//rtp_llm/models_py/distributed:collective_torch",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "main_update_compat_test",
    srcs = ["main_update_compat_test.py"],
    deps = [":new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": ""},
)

py_test(
    name = "serving_route_checkpoint_test",
    srcs = ["serving_route_checkpoint_test.py"],
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "serving_route_qwen3_test",
    srcs = [
        "serving_route_qwen3_test.py",
        "serving_route_checkpoint_test.py",
        "new_models/qwen3/main_pp_checkpoint_test.py",
    ],
    main = "serving_route_qwen3_test.py",
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "main_native_pp_checkpoint_test",
    srcs = ["new_models/qwen3/main_pp_checkpoint_test.py"],
    main = "new_models/qwen3/main_pp_checkpoint_test.py",
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
    },
    deps = [
        ":new_weight_loader",
        "//rtp_llm:transformers",
    ],
)

py_test(
    name = "eplb_direct_reload_test",
    srcs = ["eplb_direct_test.py"],
    main = "eplb_direct_test.py",
    deps = [":new_weight_loader"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    timeout = "moderate",
)

py_test(
    name = "qwen3_moe_checkpoint_test",
    srcs = [
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "qwen3_moe_checkpoint_test.py",
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "qwen3_moe_native_service_test",
    srcs = [
        "qwen3_moe_native_service_test.py",
        "qwen3_moe_checkpoint_test.py",
        "serving_route_checkpoint_test.py",
    ],
    main = "qwen3_moe_native_service_test.py",
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_test(
    name = "serving_route_deepseek_test",
    srcs = [
        "serving_route_checkpoint_test.py",
        "serving_route_deepseek_test.py",
    ],
    main = "serving_route_deepseek_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "HIP_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [":new_weight_loader", "//rtp_llm:models"],
)

py_library(
    name = "owner_migration_checkpoint_test_cases",
    testonly = True,
    srcs = ["owner_migration_checkpoint_test.py"],
    deps = [":new_weight_loader"],
    visibility = ["//rtp_llm/model_loader/test:__pkg__"],
)

py_library(
    name = "ple_residency_test_cases",
    testonly = True,
    srcs = ["new_models/qwen4_exp/ple_residency_test.py"],
    deps = [":new_weight_loader"],
    visibility = ["//rtp_llm/models/qwen4_exp/test:__pkg__"],
)

py_test(
    name = "ple_host_fst_routing_test",
    srcs = ["new_models/qwen4_exp/ple_residency_test.py"],
    main = "new_models/qwen4_exp/ple_residency_test.py",
    deps = [":new_weight_loader"],
)

py_test(
    name = "owner_migration_checkpoint_test",
    srcs = ["owner_migration_checkpoint_test.py"],
    deps = [":new_weight_loader"],
)

py_library(
    name = "utils",
    srcs = glob([
        "utils/*.py",
    ]),
    visibility = ["//visibility:public"],
    deps = ["//rtp_llm:torch"],
)

py_test(
    name = "qwen_service_vision_test",
    srcs = [
        "new_models/qwen_service_vision.py",
        "new_models/qwen_service_vision_test.py",
    ],
    main = "new_models/qwen_service_vision_test.py",
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
    deps = [
        ":new_loader_core",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:transformers",
    ],
)

# CPU-only subset of the DSv4 decode package: pure-torch modules plus the
# package __init__ chain, without the CUDA/testlib closure. The lazily
# imported paged/pool helpers (pool_slot_mapping, _pool_triton) are part of
# the closure: without them, enabling the paged or is_cuda branches under
# sandboxed runfiles raises ModuleNotFoundError instead of running the
# (pure-torch / triton-optional) helper.
py_library(
    name = "dsv4_decode_cpu",
    srcs = [
        "modules/__init__.py",
        "modules/dsv4/__init__.py",
        "modules/dsv4/_pool_triton.py",
        "modules/dsv4/decode/__init__.py",
        "modules/dsv4/decode/decode_attn_metadata.py",
        "modules/dsv4/decode/kv_write_decode_op.py",
        "modules/dsv4/fp8/__init__.py",
        "modules/dsv4/fp8/decode/__init__.py",
        "modules/dsv4/fp8/decode/pool_slot_mapping.py",
        "modules/dsv4/kv_cache_utils.py",
    ],
    data = [
        "//rtp_llm:__init__.py",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//rtp_llm:torch",
    ],
)

# flashinfer-python ships in the cuda12 and cuda13 locks
flashinfer = ["flashinfer-python"]
deep = ["deep_gemm", "deep_ep"]
flashmla = ["fast-hadamard-transform", "flash-mla", "tilelang"]
rtp_kernel = ["rtp-kernel"]
flashattn2 = ["flash_attn"]
flashattn = flashattn2 + ["flash-attn-3"]
requirement(flashinfer)
requirement(deep)
requirement(flashmla)
requirement(["flash-mla-legacy"])
requirement(rtp_kernel)
requirement(flashattn)

filegroup(
    name = "cutlass_moe_config",
    srcs = glob(["kernels/cuda/fp8_kernel/cutlass_groupgemm/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "kernel_layout_sources",
    srcs = [
        "kernels/__init__.py",
        "kernels/cuda/fp8_input_quant.py",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "rocm_fp8_ptpc_hipb_solutions",
    srcs = [
        "modules/factory/linear/impl/rocm/data/fp8_ptpc_hipb_solutions.json",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "tile_kernels_mhc",
    srcs = glob([
        "3rdparty/__init__.py",
        "3rdparty/tile_kernels/*.py",
        "3rdparty/tile_kernels/**/*.py",
    ]),
    # The mhc kernels import tilelang at module scope; carry the wheel (and
    # its z3-solver/apache-tvm-ffi closure, via the pip pkg deps) in runfiles
    # instead of relying on the image's site-packages. Only the locks used by
    # the DSv4 lanes ship tilelang.
    deps = select({
        "@//:using_cuda13_x86": ["//rtp_llm:tilelang"],
        "@//:using_cuda13_arm": ["//rtp_llm:tilelang"],
        "@//:using_cuda12_9_x86": ["//rtp_llm:tilelang"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "dsv41_jit_sources",
    srcs = glob([
        "modules/dsv41/**/*.cu",
        "modules/dsv41/**/*.cuh",
        "modules/dsv41/**/*.cc",
        "modules/dsv41/**/*.h",
        "modules/dsv41/**/*.LICENSE",
    ]),
    visibility = ["//visibility:public"],
)

_DSV4_PPU_MODULES = [
    "modules/dsv4/ppu/**/*.py",
    "modules/factory/linear/ppu_impl/fp8_linear.py",
]
_DSV4_PPU_KERNELS = [
    "kernels/ppu/attention/**/*.py", "kernels/ppu/moe/**/*.py",
    "kernels/ppu/hc/**/*.py", "kernels/ppu/mtp/**/*.py",
    "kernels/ppu/jit/**/*.py", "kernels/ppu/runtime.py",
]

cc_library(
    name = "marlin_nvfp4_kernels",
    srcs = [
        "kernels/cuda/marlin_nvfp4/marlin_kernels/schema.cpp",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/moe/marlin_moe_wna16/ops.cu",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/moe/marlin_moe_wna16/sm80_kernel_bfloat16_fe2m1f_bfloat16.cu",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu",
    ],
    hdrs = glob([
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.h",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.hpp",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/**/*.cuh",
    ]),
    includes = ["kernels/cuda/marlin_nvfp4/marlin_kernels"],
    # Keep the JIT's compilation semantics and emit both supported cubins,
    # independently of the engine's selected GPU architecture.
    copts = cuda_default_copts_without_arch() + [
        "-O3",
        "-std=c++17",
        "-DUSE_CUDA",
        # Match the runtime admission's minimum stable Torch ABI (2.10).
        "-DTORCH_TARGET_VERSION=0x020a000000000000",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "-nvcc_options=expt-relaxed-constexpr",
        "-nvcc_options=static-global-template-stub=false",
        "-nvcc_options=ftz=false",
        "--cuda-gpu-arch=sm_90",
        "--cuda-gpu-arch=sm_120",
    ],
    alwayslink = True,
    target_compatible_with = select({
        "//:using_cuda13_x86": [],
        "//:using_cuda13_arm": [],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    deps = select({
        "//:using_cuda13_x86": ["@torch_2.11_py310_cuda//:torch"],
        "//:using_cuda13_arm": ["@torch_2.11_py310_cuda-aarch64//:torch"],
        "//conditions:default": [],
    }) + [
        "@local_config_cuda//cuda:cuda_headers",
        "@local_config_cuda//cuda:cudart",
    ],
)

cc_binary(
    name = "marlin_nvfp4",
    linkopts = [
        "-Wl,--as-needed",
        "-Wl,-rpath='$$ORIGIN/../../torch/lib'",
    ],
    linkshared = True,
    deps = [":marlin_nvfp4_kernels"],
    visibility = ["//visibility:public"],
)

py_library(
    name = "modules",
    srcs = glob([
        "modules/*.py",
        "modules/**/*.py",
    ], exclude = _DSV4_PPU_MODULES + ["modules/qwen4_exp/test/**"]) + select({
        "@//:using_ppu": glob(_DSV4_PPU_MODULES),
        "//conditions:default": [],
    }),
    data = [
        ":cutlass_moe_config",
        ":dsv41_jit_sources",
        ":rocm_fp8_ptpc_hipb_solutions",
    ] + select({
        "@//:using_ppu": [":ppu_sources"],
        "//conditions:default": [],
    }),
    deps = [
        ":utils",
        ":kernels",
        ":tile_kernels_mhc",
        "//rtp_llm:warmup",
        "//rtp_llm/models_py/triton_kernels:triton_kernels",
        "//rtp_llm/models_py/distributed:deepep_wrapper",
        "//rtp_llm/models_py/distributed:moriep_wrapper",
        "//rtp_llm/models_py/distributed:collective_torch",
        "//rtp_llm/models_py/distributed:cpu_status",
        "//rtp_llm/models_py/distributed:rank_layout",
        "//rtp_llm/models_py/distributed:user_buffers",
    ] + select({
        # cuda13 locks ship flashinfer-python and deep-ep.
        "@//:using_cuda13_x86": flashinfer + deep,
        "@//:using_cuda13_arm": flashinfer + deep,
        "@//:using_cuda12": flashinfer + deep,
        "@//:using_ppu": deep,
        "@//:using_rocm": [
            "//rtp_llm/models_py/modules/base/rocm:rocm_base",
            "//rtp_llm/models_py/modules/base/rocm:trt_allreduce",
        ],
        "//conditions:default": [],
    }) + select({
        # cuda12_arm (kept from main) has no branch here: its lock ships none of these.
        "@//:using_cuda13_x86": ["flash-mla", "fast-hadamard-transform", "tilelang"],
        "@//:using_cuda13_arm": ["flash-mla", "flash-mla-legacy", "tilelang"],
        "@//:using_cuda12_9_x86": flashmla,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda13_x86": rtp_kernel,
        "@//:using_cuda13_arm": rtp_kernel,
        "@//:using_cuda12_9_x86": rtp_kernel,
        "@//:using_cuda12_arm": rtp_kernel,
        "//conditions:default": [],
    }) + select({
        "@//:using_cuda13_x86": flashattn2,
        "@//:using_cuda13_arm": [],
        "@//:using_cuda12_9_x86": flashattn,
        "@//:using_cuda12_arm": flashattn,
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

py_library(
    name = "dsv4_fused_kernel_sources",
    srcs = [
        "modules/dsv4/_fused_inv_rope_fp8_quant_triton.py",
        "modules/dsv4/_fused_rmsnorm_rope_triton.py",
        "modules/dsv4/_rope_only_triton.py",
    ],
    visibility = ["//visibility:public"],
)

# Export sitecustomize.py for test targets to fix nvidia-cutlass-dsl import
exports_files(["modules/factory/sitecustomize.py"])

exports_files(
    [
        "modules/dsv4/ppu/resources.py",
        "modules/dsv4/ppu/moe/ppu_ep_moe.py",
        "modules/dsv4/ppu/metadata/cache.py",
        "modules/dsv4/ppu/manifest.py",
    ],
    visibility = ["//rtp_llm/models_py/modules/dsv4/ppu/test:__pkg__"],
)

py_library(
    name = "modules_cuda",
    srcs = glob([
        "modules/cuda/*.py",
        "modules/cuda/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "modules_rocm",
    srcs = glob([
        "modules/rocm/*.py",
        "modules/rocm/**/*.py",
    ]),
    visibility = [":__subpackages__"],
)

py_library(
    name = "ppu_entrypoint",
    srcs = [
        "__init__.py",
        "modules/factory/attention/register.py",
        "modules/factory/fused_moe/register.py",
        "modules/factory/linear/register.py",
    ],
    deps = [
        "//rtp_llm:device",
        "//rtp_llm:utils",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "ppu_modules",
    # Keep the public PPU test closure aligned with the internal ppu_modules
    # target: the factory imports the shared model runtime during collection.
    srcs = [],
    deps = [
        ":ppu_entrypoint",
        ":modules",
        "//rtp_llm:models",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:ops",
        "//rtp_llm:lora",
        "//rtp_llm:transformers",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm:pydantic",
        "//rtp_llm:utils",
    ],
    data = [
        "//:rtp_compute_ops",
        "//:th_transformer",
        "//rtp_llm/libs:libth_transformer_config_so",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "scaled_rope",
    srcs = ["kernels/cuda/scaled_rope.py"],
    deps = ["//rtp_llm:torch"] + triton_deps(["triton"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "kernels",
    srcs = glob([
        "kernels/*.py",
        "kernels/**/*.py",
    ], exclude = _DSV4_PPU_KERNELS + ["kernels/**/test/**", "kernels/cuda/scaled_rope.py"]) + select({
        "@//:using_ppu": glob(_DSV4_PPU_KERNELS),
        "//conditions:default": [],
    }),
    data = [
        "//rtp_llm/libs:marlin_nvfp4",
        "kernels/cuda/marlin_nvfp4/marlin_kernels/LICENSE",
    ] + glob(["kernels/cuda/qsa/*.cu", "kernels/cuda/qsa/*.cuh"]),
    deps = [
        ":scaled_rope",
        ":utils",
        "//rtp_llm:torch",
        "//rtp_llm/models_py/triton_kernels:common",
        "//rtp_llm/models_py/triton_kernels:fla",
    ] + triton_deps(["triton"]),
    visibility = ["//rtp_llm/models_py:__subpackages__"],
)

py_library(
    name = "internal_kernels",
    srcs = glob(["kernels/ppu/**/*.py"], exclude = _DSV4_PPU_KERNELS) + select({
        "@//:using_ppu": glob(_DSV4_PPU_KERNELS),
        "//conditions:default": [],
    }),
    data = select({
        "@//:using_ppu": [":ppu_jit_sources"],
        "//conditions:default": [],
    }),
    deps = [
        "//rtp_llm:torch",
        "//rtp_llm:triton",
    ],
    visibility = ["//visibility:public"],
)

py_library(
    name = "configs",
    srcs = glob([
        "configs/*.py",
    ]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "models",
    srcs = glob([
        "model_desc/*.py",
        "model_desc/cuda_graph/*.py",
        "model_desc/qwen4_exp_runtime/*.py",
        "speculative/*.py",
    ]) + select({
        "@//:using_ppu": glob(["model_desc/deepseek_v4_ppu/**/*.py"]),
        "//conditions:default": [],
    }),
    deps = [
        ":utils",
        ":modules",
        ":kernels",
        ":configs",
        "//rtp_llm:config",
        "//rtp_llm:device",
        "//rtp_llm:warmup",
        "//rtp_llm/model_loader:loader",
    ] + select({
        "@//:using_cuda12": [
            ":modules_cuda",
        ],
        "@//:using_rocm": [
            ":modules_rocm"
        ],
        "//conditions:default": [],
    }) + select({
        # deepseek_v4_model.py lazily imports modules/dsv4/tilelang_kernels and
        # the DSv4 warmup reaches the mhc TileLang kernels; declare tilelang on
        # the owning library so its runfiles do not depend on the image's
        # site-packages install. z3-solver/apache-tvm-ffi ride along through the
        # tilelang pip pkg deps.
        "@//:using_cuda13_x86": ["//rtp_llm:tilelang"],
        "@//:using_cuda13_arm": ["//rtp_llm:tilelang"],
        "@//:using_cuda12_9_x86": ["//rtp_llm:tilelang"],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
)

# PPU-native JIT sources retain their frozen relative include layout.
filegroup(
    name = "ppu_jit_sources",
    srcs = glob(["kernels/ppu/**/*.cuh", "kernels/ppu/**/*.h", "kernels/ppu/**/*.json", "kernels/ppu/**/*.patch", "kernels/ppu/**/NOTICE*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ppu_sources",
    srcs = glob([
        "model_desc/deepseek_v4_ppu/**/*.py",
        "modules/dsv4/ppu/**/*.py",
        "modules/factory/linear/ppu_impl/*.py",
        "modules/factory/fused_moe/ppu_impl/*.py",
        "kernels/ppu/**/*.py",
    ], exclude = ["**/test/**", "**/tests/**"]) + [":ppu_jit_sources", "//rtp_llm:models/deepseek_v4.py"],
    visibility = ["//visibility:public"],
)
