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
    deps = ["//tensorflow/Source/lm:lm"],
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
