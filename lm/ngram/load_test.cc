#include "load.h"
#include "tensorflow/core/platform/test.h"
#include <fstream>


void SetUp(NGram *under_test) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/load_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the mouse .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();

    under_test->ProcessFile(test_file_name);
}

TEST(LoadTest, NGram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, AddOne) {
    NGram *under_test = new AddOne(3, 1);
    ::SetUp(under_test);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}

TEST(LoadTest, KneserNey) {
    NGram *under_test = new KneserNey(3, 0.5, 1);
    ::SetUp(under_test);
    under_test->Save("/tmp");

    NGram *under_test_loaded = Load("/tmp");

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
