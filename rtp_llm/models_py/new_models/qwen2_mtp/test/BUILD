py_test(
    name = "test_qwen2_mtp_load",
    srcs = ["test_qwen2_mtp_load.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:safetensors_torch",
        "//rtp_llm:sdk",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
