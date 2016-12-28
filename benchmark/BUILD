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
