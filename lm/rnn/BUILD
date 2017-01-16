py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
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
    srcs = [
        "lstm.h",
        "lstm.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/core:tensorflow",
        "//tensorflow/Source/lm:lm",
        "//tensorflow/Source/lm:vocab",
    ]
)

py_binary(
    name = "ptb_word_lm",
    srcs = ["ptb_word_lm.py"],
    deps = [
        ":ptb_reader",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "ptb_reader",
    srcs = ["ptb_reader.py"],
    deps = [],
)
