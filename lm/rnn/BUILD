py_binary(
    name = "lstm",
    srcs = ["lstm.py"],
    deps = [
        ":reader",
        "//tensorflow/Source/lm:vocab_pb2",
        "//tensorflow:tensorflow_py",
    ],
)

py_library(
    name = "reader",
    srcs = ["reader.py"],
    deps = [],
)
