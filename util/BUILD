cc_library(
    name = "reader",
    srcs = ["reader.cc"],
    hdrs = ["reader.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

cc_test(
    name = "reader_test",
    size = "small",
    srcs = ["reader_test.cc"],
    deps = [
        ":reader",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
