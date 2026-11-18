py_test(
    name = "qwen7b_pd_checkpoint_test",
    srcs = ["qwen7b_pd_checkpoint_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    deps = [
        "//rtp_llm:model_factory",
        "//rtp_llm:fastsafetensors",
        "//rtp_llm:safetensors_torch",
        "//rtp_llm/models_py:new_weight_loader",
    ],
)
