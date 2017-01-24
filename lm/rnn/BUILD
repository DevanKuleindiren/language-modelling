load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

py_binary(
    name = "rnn",
    srcs = ["rnn.py"],
    deps = [
        ":reader",
        ":rnn_pb_py",
        "//tensorflow/Source/lm:vocab_pb_py",
        "//tensorflow:tensorflow_py",
    ],
)

tf_proto_library(
    name = "rnn_pb",
    srcs = ["rnn.proto"],
    cc_api_version = 2,
    py_api_version = 2,
    visibility = ["//visibility:public"],
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
    name = "rnn_inference",
    srcs = ["rnn.cc"],
    hdrs = ["rnn.h"],
    visibility = ["//visibility:public"],
    deps = [
        ":rnn_pb_cc",
        "//tensorflow/core:tensorflow",
        "//tensorflow/Source/lm:lm",
        "//tensorflow/Source/lm:vocab",
    ]
)

cc_test(
    name = "rnn_inference_test",
    size = "small",
    srcs = ["rnn_test.cc"],
    deps = [
        ":rnn_inference",
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
        "test_data/rnn.pbtxt",
        "test_data/vocab.pbtxt",
    ],
)

py_binary(
    name = "gen_data",
    srcs = ["test_data/gen_data.py"],
    deps = [
        ":rnn_pb_py",
        "//tensorflow/Source/lm:vocab_pb_py",
        "//tensorflow:tensorflow_py",
    ],
)
