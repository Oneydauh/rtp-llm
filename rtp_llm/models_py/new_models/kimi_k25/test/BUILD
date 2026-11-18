py_test(
    name = "test_kimi_gpu",
    srcs = ["test_kimi_gpu.py", "test_kimi_vision.py", "test_kimi_compressed.py"],
    tags = ["H20"],
    exec_properties = {"gpu": "H20"},
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_language",
    srcs = ["test_language.py"],
    deps = [
        "//rtp_llm:models",
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_kimi_vision",
    srcs = ["test_kimi_vision.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)

py_test(
    name = "test_kimi_compressed",
    srcs = ["test_kimi_compressed.py"],
    deps = [
        "//rtp_llm:safetensors",
        "//rtp_llm:torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
