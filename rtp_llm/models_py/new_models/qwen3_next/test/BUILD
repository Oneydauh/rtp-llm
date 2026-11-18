py_test(
    name = "gdn_parameters_test",
    srcs = ["gdn_parameters_test.py"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)

py_test(
    name = "main_checkpoint_test",
    srcs = ["main_checkpoint_test.py"],
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = ["//rtp_llm/models_py:new_weight_loader"],
)
