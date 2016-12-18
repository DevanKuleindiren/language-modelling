cc_library(
    name = "lm",
    srcs = ["lm.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "vocab_pb",
    srcs = ["vocab.proto"],
    cc_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)
