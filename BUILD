cc_binary(
    name = "main",
    srcs = ["main.cc"],
    deps = [
        "//tensorflow/core:core_cpu",
        "//tensorflow/core:lib",
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:framework_headers_lib",
        "//tensorflow/Source/benchmark:benchmark",
        "//tensorflow/Source/lm/ngram:load",
        "//tensorflow/Source/lm/ngram/smoothing:add_one",
        "//tensorflow/Source/lm/ngram/smoothing:kneser_ney",
        "//tensorflow/Source/lm/rnn:lstm_inference",
    ],
)
