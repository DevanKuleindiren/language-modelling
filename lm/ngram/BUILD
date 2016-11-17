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
        "//lm:lm",
    ],
)

cc_test(
    name = "ngram_test",
    size = "small",
    srcs = ["ngram_test.cc"],
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":ngram",
    ],
)

cc_library(
    name = "prob_trie",
    srcs = [
        "prob_trie.h",
        "prob_trie.cc",
    ],
    deps = [],
)

cc_test(
    name = "prob_trie_test",
    size = "small",
    srcs = ["prob_trie_test.cc"],
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":prob_trie",
    ],
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
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":count_trie",
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
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":vocab",
    ],
)
