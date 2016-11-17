#include "gtest/gtest.h"
#include "count_trie.h"


class CountTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new CountTrie(3);
        vocab = new Vocab();

        std::ofstream test_file;
        std::string test_file_name = "/tmp/count_trie_test_file";
        test_file.open (test_file_name);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test->ProcessFile(test_file_name, vocab);

        /*
        When the file is processed, the words are mapped to integers. In this test, we have:
        <s>   -> 0
        the   -> 1
        cat   -> 2
        sat   -> 3
        on    -> 4
        mat   -> 5
        .     -> 6
        ate   -> 7
        mouse -> 8
        dog   -> 9
        */
    }

    CountTrie *under_test;
    Vocab *vocab;
};

TEST_F(CountTrieTest, Count) {
    EXPECT_EQ(under_test->Count(std::list<size_t>({})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({10})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({10, 10})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({10, 10, 10})), 0);
    EXPECT_EQ(under_test->Count(std::list<size_t>({1})), 6);
    EXPECT_EQ(under_test->Count(std::list<size_t>({2})), 3);
    EXPECT_EQ(under_test->Count(std::list<size_t>({3})), 2);
    EXPECT_EQ(under_test->Count(std::list<size_t>({8})), 1);
    EXPECT_EQ(under_test->Count(std::list<size_t>({1, 2})), 3);
    EXPECT_EQ(under_test->Count(std::list<size_t>({5, 6})), 1);
    EXPECT_EQ(under_test->Count(std::list<size_t>({3, 4, 1})), 2);
    EXPECT_EQ(under_test->Count(std::list<size_t>({3, 4, 2})), 0);
}

TEST_F(CountTrieTest, CountFollowing) {
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({})), 10);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 10})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 10, 10})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({1})), 4);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({3})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({7})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({6})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({1, 2})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({4, 1})), 2);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({9, 3})), 1);
}

TEST_F(CountTrieTest, CountPreceding) {
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({10})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({10, 10})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({10, 10, 10})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({1})), 3);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({2})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({6})), 3);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({0})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({1, 2})), 2);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({8, 6})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<size_t>({9, 1})), 0);
}

TEST_F(CountTrieTest, CountPrecedingAndFollowing) {
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({})), 14);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({10})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({10, 10})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({10, 10, 10})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({1})), 5);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({2})), 3);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({7})), 1);
}

TEST_F(CountTrieTest, SumFollowing) {
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({})), 20);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({10})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({10, 10})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({10, 10, 10})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({1})), 6);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({2})), 3);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({3})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({8})), 1);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({6})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({4, 1})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({3, 4})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({1, 2})), 3);
}

TEST_F(CountTrieTest, VocabSize) {
    EXPECT_EQ(under_test->VocabSize(), 9);
}
