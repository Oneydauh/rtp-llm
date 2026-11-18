py_test(
    name = "test_fp8_already_quantized",
    srcs = ["test_fp8_already_quantized.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_block_fp8_ft",
    srcs = ["test_block_fp8_ft.py"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_glm_ft_block_tree",
    srcs = ["test_glm_ft_block_tree.py", "test_block_fp8_ft.py"],
    main = "test_glm_ft_block_tree.py",
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm:torch",
    ],
)

py_test(
    name = "test_glm_ft_block_serving",
    srcs = ["test_glm_ft_block_serving.py", "test_glm_ft_block_tree.py", "test_block_fp8_ft.py"],
    main = "test_glm_ft_block_serving.py",
    data = ["//:rtp_compute_ops", "//:th_transformer_config"],
    deps = [
        "//rtp_llm/models_py:new_weight_loader",
        "//rtp_llm/tools/convert:convert",
    ],
    exec_properties = {"gpu": "H20"},
    tags = ["H20"],
)
