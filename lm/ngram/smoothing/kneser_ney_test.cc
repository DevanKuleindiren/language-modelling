#include "gtest/gtest.h"
#include "kneser_ney.h"
#include <fstream>


class KneserNeyTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new KneserNey(3, 0.5);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/ngram_test_file";
        test_file.open (test_file_name);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test->ProcessFile(test_file_name);
    }
    KneserNey *under_test;
};

TEST_F(KneserNeyTest, Prob) {
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 64/105.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 2/7.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 11/35.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 1/28.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 45/56.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}
