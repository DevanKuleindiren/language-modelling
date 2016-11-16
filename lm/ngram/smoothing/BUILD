cc_library(
    name = "add_one",
    srcs = [
        "add_one.h",
        "add_one.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//lm/ngram:ngram",
    ],
)

cc_test(
    name = "add_one_test",
    size = "small",
    srcs = ["add_one_test.cc"],
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":add_one",
    ],
)

cc_library(
    name = "kneser_ney",
    srcs = [
        "kneser_ney.h",
        "kneser_ney.cc",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//lm/ngram:ngram",
    ],
)

cc_test(
    name = "kneser_ney_test",
    size = "small",
    srcs = ["kneser_ney_test.cc"],
    copts = ["-Iexternal/gtest/googletest-release-1.8.0/googletest/include"],
    deps = [
        "@gtest//:main",
        ":kneser_ney",
    ],
)
