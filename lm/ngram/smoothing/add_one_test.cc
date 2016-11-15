#include "gtest/gtest.h"
#include "add_one.h"
#include <fstream>


class AddOneTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new AddOne(3);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/ngram_test_file";
        test_file.open (test_file_name);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test->ProcessFile(test_file_name);
    }
    AddOne *under_test;
};

TEST_F(AddOneTest, Prob) {
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 0.25);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/6.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 2/11.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 0.1);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 1/9.0);
}
