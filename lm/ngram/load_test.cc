#include "load.h"
#include "tensorflow/core/platform/test.h"
#include <fstream>


std::string SetUp() {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/load_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the mouse .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();

    return test_file_name;
}

TEST(LoadTest, NGram) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, AbsoluteDiscounting) {
    NGram *under_test = new AbsoluteDiscounting(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, AddOne) {
    NGram *under_test = new AddOne(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, Katz) {
    NGram *under_test = new Katz(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, KneserNey) {
    NGram *under_test = new KneserNey(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, KneserNeyMod) {
    NGram *under_test = new KneserNeyMod(::SetUp(), 3, 1);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
