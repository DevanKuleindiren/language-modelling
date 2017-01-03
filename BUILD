cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/Source/lm/ngram:load",
        "//tensorflow/Source/lm/rnn:lstm_inference",
    ],
)
