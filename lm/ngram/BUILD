py_library(
    name = "ngram",
    srcs = [
        "model.py",
        "trie.py",
    ],
    visibility = ["//visibility:public"],
    deps = []
)

py_test(
    name = "trie_test",
    size = "small",
    srcs = ["trie_test.py"],
    deps = ["//lm/ngram:ngram"],
)