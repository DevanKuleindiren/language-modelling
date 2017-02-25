cc_library(
    name = "char_trie",
    srcs = ["char_trie.cc"],
    hdrs = ["char_trie.h"],
    visibility = ["//visibility:public"],
    deps = [":prediction_compare"],
)

cc_test(
    name = "char_trie_test",
    size = "small",
    srcs = ["char_trie_test.cc"],
    deps = [
        ":char_trie",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "prediction_compare",
    srcs = ["prediction_compare.h"],
    visibility = ["//visibility:public"],
    deps = [],
)

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

cc_library(
    name = "dual_reader",
    hdrs = ["dual_reader.h"],
    visibility = ["//visibility:public"],
    deps = [":reader"],
)

cc_test(
    name = "dual_reader_test",
    size = "small",
    srcs = ["dual_reader_test.cc"],
    deps = [
        ":dual_reader",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

py_binary(
    name = "tokenise",
    srcs = ["tokenise.py"],
    deps = [],
)

py_binary(
    name = "count_same",
    srcs = ["count_same.py"],
    deps = [],
)
