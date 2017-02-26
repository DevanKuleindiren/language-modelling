cc_library(
    name = "error_rnn",
    srcs = ["error_rnn.cc"],
    hdrs = ["error_rnn.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/Source/lm:vocab",
        "//tensorflow/Source/lm/rnn:rnn_inference",
        "//tensorflow/Source/lm/rnn:rnn_pb_cc",
        "//tensorflow/Source/util:char_trie",
        "//tensorflow/Source/util:prediction_compare",
    ]
)

cc_test(
    name = "error_rnn_test",
    size = "small",
    srcs = ["error_rnn_test.cc"],
    deps = [
        ":error_rnn",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
    data = [
        "//tensorflow/Source/lm/rnn:test_data",
    ],
)
