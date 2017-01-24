py_binary(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        ":reader",
        "//tensorflow/Source/lm:vocab_pb_py",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = [],
)

py_test(
    name = "reader_test",
    size = "small",
    srcs = ["reader_test.py"],
    deps = [
        ":reader",
        "//tensorflow:tensorflow_py",
    ],
)

cc_library(
    name = "lstm_inference",
    srcs = ["lstm.cc"],
    hdrs = ["lstm.h"],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/Source/lm:lm",
        "//tensorflow/Source/lm:vocab",
    ]
)

cc_test(
    name = "lstm_inference_test",
    size = "small",
    srcs = ["lstm_test.cc"],
    deps = [
        ":lstm_inference",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
    data = [
        ":test_data",
    ],
)

filegroup(
    name = "test_data",
    srcs = [
        "test_data/graph.pb",
        "test_data/vocab.pbtxt",
    ],
)

py_binary(
    name = "gen_data",
    srcs = ["test_data/gen_data.py"],
    deps = [
        "//tensorflow/Source/lm:vocab_pb_py",
        "//tensorflow:tensorflow_py",
    ],
)
