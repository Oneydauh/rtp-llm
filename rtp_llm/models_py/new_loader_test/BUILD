py_test(
    name = "foundation_test",
    srcs = ["foundation_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "foundation_gpu_test",
    srcs = ["foundation_gpu_test.py"],
    exec_properties = {
        "gpu": "H20",
        "gpu_count": "1",
    },
    tags = ["H20"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:numpy",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "read_spec_test",
    srcs = ["read_spec_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "runtime_weight_update_test",
    srcs = ["runtime_weight_update_test.py"],
    deps = ["//rtp_llm/models_py:new_loader_core", "//rtp_llm:torch"],
)

py_test(
    name = "runtime_weight_update_gpu_test",
    srcs = ["runtime_weight_update_test.py"],
    main = "runtime_weight_update_test.py",
    exec_properties = {"gpu": "H20", "gpu_count": "1"},
    tags = ["H20"],
    deps = ["//rtp_llm/models_py:new_loader_core", "//rtp_llm:torch"],
)

py_test(
    name = "new_weight_manager_test",
    srcs = ["new_weight_manager_test.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/model_loader:loader",
        "//rtp_llm/models_py/new_models/qwen3/test:qwen3_dense_load_test_lib",
        "//rtp_llm:torch",
    ],
)
