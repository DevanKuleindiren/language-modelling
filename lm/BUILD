cc_library(
    name = "lm",
    srcs = ["lm.h"],
    visibility = [
        "//tensorflow/Source/lm/ngram:__subpackages__",
        "//tensorflow/Source/benchmark:__subpackages__"],
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
