package(default_visibility = ["//visibility:public"])

cc_library(
    name = "ngram",
    srcs = [
        "ngram.h",
        "ngram.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":count_trie",
        ":prob_trie",
        "//tensorflow/Source/lm:lm",
    ],
)

cc_test(
    name = "ngram_test",
    size = "small",
    srcs = ["ngram_test.cc"],
    deps = [
        ":ngram",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "prob_trie",
    srcs = [
        "prob_trie.h",
        "prob_trie.cc",
    ],
    deps = [":prob_trie_pb_cc"],
)

cc_test(
    name = "prob_trie_test",
    size = "small",
    srcs = ["prob_trie_test.cc"],
    deps = [
        ":prob_trie",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

load(
    "//tensorflow/core:platform/default/build_config.bzl",
    "tf_proto_library",
)

tf_proto_library(
    name = "prob_trie_pb",
    srcs = ["prob_trie.proto"],
    cc_api_version = 2,
    visibility = ["//visibility:public"],
)

cc_library(
    name = "count_trie",
    srcs = [
        "count_trie.h",
        "count_trie.cc",
    ],
    deps = [":vocab"],
)

cc_test(
    name = "count_trie_test",
    size = "small",
    srcs = ["count_trie_test.cc"],
    deps = [
        ":count_trie",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)

cc_library(
    name = "vocab",
    srcs = [
        "vocab.h",
        "vocab.cc",
    ],
    deps = [],
)

cc_test(
    name = "vocab_test",
    size = "small",
    srcs = ["vocab_test.cc"],
    deps = [
        ":vocab",
        "//tensorflow/core:test",
        "//tensorflow/core:test_main",
    ],
)
