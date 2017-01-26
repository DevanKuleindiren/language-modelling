load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

cc_library(
    name = "lm",
    srcs = ["lm.cc"],
    hdrs = ["lm.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":vocab",
        "//tensorflow/Source/util:prediction_compare",
    ],
)

cc_library(
    name = "vocab",
    srcs = [
        "vocab.h",
        "vocab.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [":vocab_pb_cc"],
)

cc_test(
    name = "vocab_test",
    size = "small",
    srcs = ["vocab_test.cc"],
    deps = [
        ":vocab",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_proto_library(
    name = "vocab_pb",
    srcs = ["vocab.proto"],
    cc_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
)
