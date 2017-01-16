#include "tensorflow/core/platform/test.h"
#include "vocab.h"


class VocabTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new Vocab(2);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/vocab_test_file";
        test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
        test_file << " the cat sat on the mat . \n";
        test_file << " the cat ate the mouse . \n";
        test_file << " the dog sat on the cat . \n";
        test_file.close();

        under_test->ProcessFile(test_file_name);
    }

    Vocab *under_test;
};

void SetUpTestFileWithString(std::string test_file_name, std::string content) {
    std::ofstream test_file;
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << content;
    test_file.close();
}

TEST_F(VocabTest, IdsCorrect) {
    EXPECT_EQ(under_test->Get("<unk>"), 0);
    EXPECT_EQ(under_test->Get("blah"), 0);
    EXPECT_EQ(under_test->Get("<s>"), 1);
    EXPECT_EQ(under_test->Get("the"), 2);
    EXPECT_EQ(under_test->Get("cat"), 3);
    EXPECT_EQ(under_test->Get("sat"), 5);
    EXPECT_EQ(under_test->Get("on"), 6);
    EXPECT_EQ(under_test->Get("mat"), 0);
    EXPECT_EQ(under_test->Get("."), 4);
    EXPECT_EQ(under_test->Get("ate"), 0);
    EXPECT_EQ(under_test->Get("mouse"), 0);
    EXPECT_EQ(under_test->Get("dog"), 0);
}

TEST_F(VocabTest, Iterator) {
    std::unordered_map<std::string, size_t> actual_output;
    std::unordered_map<std::string, size_t> expected_output;
    expected_output.insert(std::make_pair("<unk>", 0));
    expected_output.insert(std::make_pair("<s>", 1));
    expected_output.insert(std::make_pair("the", 2));
    expected_output.insert(std::make_pair("cat", 3));
    expected_output.insert(std::make_pair(".", 4));
    expected_output.insert(std::make_pair("sat", 5));
    expected_output.insert(std::make_pair("on", 6));

    std::unordered_map<std::string, size_t>::const_iterator it = under_test->begin();
    for (; it != under_test->end(); ++it) {
        actual_output.insert(*it);
    }

    EXPECT_EQ(actual_output, expected_output);
}

TEST(VocabTestOps, EqualsOpTrue) {
    Vocab *vocab_a = new Vocab(2);
    Vocab *vocab_b = new Vocab(2);

    std::string test_file_name = "/tmp/vocab_test_file";
    ::SetUpTestFileWithString(test_file_name, "the cat sat on the mat .\nthe cat ate the mouse .\n");

    vocab_a->ProcessFile(test_file_name);
    vocab_b->ProcessFile(test_file_name);

    ASSERT_TRUE(*vocab_a == *vocab_b);
}

TEST(VocabTestOps, EqualsOpFalseMinFreq) {
    Vocab *vocab_a = new Vocab(1);
    Vocab *vocab_b = new Vocab(2);

    std::string test_file_name = "/tmp/vocab_test_file";
    ::SetUpTestFileWithString(test_file_name, "the cat sat on the mat .\nthe cat ate the mouse .\n");

    vocab_a->ProcessFile(test_file_name);
    vocab_b->ProcessFile(test_file_name);

    ASSERT_FALSE(*vocab_a == *vocab_b);
}

TEST(VocabTestOps, EqualsOpFalseWords) {
    Vocab *vocab_a = new Vocab(1);
    Vocab *vocab_b = new Vocab(2);

    std::string test_file_name_a = "/tmp/vocab_test_file_a";
    ::SetUpTestFileWithString(test_file_name_a, "the cat sat on the mat .\nthe cat ate the mouse .\n");
    std::string test_file_name_b = "/tmp/vocab_test_file_b";
    ::SetUpTestFileWithString(test_file_name_b, "the cat ate the mouse .\nthe cat sat on the mat .\n");

    vocab_a->ProcessFile(test_file_name_a);
    vocab_b->ProcessFile(test_file_name_b);

    ASSERT_FALSE(*vocab_a == *vocab_b);
}

TEST_F(VocabTest, SaveAndLoadEqual) {
    std::string file_name = "/tmp/vocab_test_file.pbtxt";
    under_test->Save(file_name);
    ASSERT_TRUE(*under_test == *Vocab::Load(file_name));
}
