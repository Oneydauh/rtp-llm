package(default_visibility = ["//:__subpackages__"])

py_library(
    name = "loader",
    srcs = glob([
        "*.py",
    ]),
    deps = [
        "//rtp_llm:lora",
        "//rtp_llm:utils",
        "//rtp_llm/models_py:new_loader_core",
        "//rtp_llm/model_loader/tipc:tipc"
    ]
)
