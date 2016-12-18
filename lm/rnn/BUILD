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

cc_library(
    name = "lstm_inference",
    srcs = [
        "lstm.h",
        "lstm.cc"
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//tensorflow/Source/lm:lm",
        "//tensorflow/Source/lm/ngram:vocab",
        "//tensorflow/core:tensorflow",
    ]
)
