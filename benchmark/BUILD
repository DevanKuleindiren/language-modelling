load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

cc_library(
    name = "benchmark",
    srcs = [
        "benchmark.h",
        "benchmark.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/Source/lm:lm",
        "//tensorflow/Source/util:reader",
    ],
)

cc_test(
    name = "benchmark_test",
    size = "small",
    srcs = ["benchmark_test.cc"],
    deps = [
        ":benchmark",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

tf_proto_library(
    name = "benchmark_pb",
    srcs = ["benchmark.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        ":benchmark",
        ":benchmark_pb_cc",
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/Source/lm/ngram:load",
        "//tensorflow/Source/lm/rnn:rnn_inference",
    ],
)
