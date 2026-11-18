py_test(
    name = "test_inputs",
    srcs = ["test_inputs.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "test_language",
    srcs = ["test_language.py"],
    env = {
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "HIP_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
    },
    deps = [
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors",
        "//rtp_llm:sdk",
        "//rtp_llm:torch",
        "//rtp_llm:transformers",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
