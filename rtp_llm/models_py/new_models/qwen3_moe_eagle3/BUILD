py_library(
    name = "draft_body",
    srcs = ["__init__.py", "body.py", "proposal.py", "service.py"],
    deps = ["//rtp_llm/models_py:new_loader_core", "//rtp_llm:torch"],
    visibility = ["//visibility:public"],
)

py_test(
    name = "draft_body_test",
    srcs = ["draft_body_test.py"],
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":draft_body", "//rtp_llm:safetensors"],
)

py_test(
    name = "proposal_test",
    srcs = ["draft_body_test.py", "proposal_test.py"],
    main = "proposal_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    deps = [":draft_body", "//rtp_llm:safetensors"],
)

py_test(
    name = "service_test",
    srcs = ["draft_body_test.py", "service_test.py"],
    main = "service_test.py",
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NEWLOADER_TEST_NATIVE_FST": "1"},
    deps = [":draft_body", "//rtp_llm:models", "//rtp_llm:sdk", "//rtp_llm:transformers", "//rtp_llm/models_py:new_weight_loader", "//rtp_llm:safetensors"],
)
