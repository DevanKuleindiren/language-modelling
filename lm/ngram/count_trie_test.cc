#include "count_trie.h"
#include "tensorflow/core/platform/test.h"


class CountTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new CountTrie(3);
        vocab = new Vocab(1);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/count_trie_test_file";
        test_file.open (test_file_name);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        vocab->ProcessFile(test_file_name);
        under_test->ProcessFile(test_file_name, vocab);

        /*
        When the file is processed, the words are mapped to integers. In this test, we have:
        <unk> -> 0
        <s>   -> 1
        the   -> 2
        cat   -> 3
        sat   -> 4
        on    -> 5
        mat   -> 6
        .     -> 7
        ate   -> 8
        mouse -> 9
        dog   -> 10
        */
    }

    CountTrie *under_test;
    Vocab *vocab;
};

TEST_F(CountTrieTest, Count) {
    EXPECT_EQ(under_test->Count(std::list<size_t>({})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({2})), 6);
    EXPECT_EQ(under_test->Count(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->Count(std::list<size_t>({4})), 2);
    EXPECT_EQ(under_test->Count(std::list<size_t>({9})), 1);
    EXPECT_EQ(under_test->Count(std::list<size_t>({2, 3})), 3);
    EXPECT_EQ(under_test->Count(std::list<size_t>({6, 7})), 1);
    EXPECT_EQ(under_test->Count(std::list<size_t>({4, 5, 2})), 2);
    EXPECT_EQ(under_test->Count(std::list<size_t>({4, 5, 3})), 0);
}

TEST_F(CountTrieTest, CountFollowing) {
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({})), 10);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2})), 4);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({4})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({8})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({7})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2, 3})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({5, 2})), 2);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 4})), 1);
}

TEST_F(CountTrieTest, CountPreceding) {
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({2})), 3);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({3})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({7})), 3);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({0})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({2, 3})), 2);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({9, 7})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({10, 2})), 0);
}

TEST_F(CountTrieTest, CountPrecedingAndFollowing) {
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({})), 14);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({2})), 5);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({8})), 1);
}

TEST_F(CountTrieTest, SumFollowing) {
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({})), 20);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({2})), 6);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({4})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({9})), 1);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({7})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({5, 2})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({4, 5})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({2, 3})), 3);
}

TEST_F(CountTrieTest, VocabSize) {
    EXPECT_EQ(under_test->VocabSize(), 9);
}
