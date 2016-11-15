#include "gtest/gtest.h"
#include "count_trie.h"


class CountTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new CountTrie(3);
        std::list<std::string> test_words({"the", "cat", "sat", "on", "the", "mat", "sat", "on", "the", "floor", "."});
        std::list<std::string> window;

        for (auto forward_it = test_words.begin(); forward_it != test_words.end(); forward_it++) {
            window.push_back(*forward_it);
            if (window.size() > 3) {
                window.pop_front();
            }
            std::list<std::string> ngram;
            for (auto reverse_it = window.rbegin(); reverse_it != window.rend(); reverse_it++) {
                ngram.push_front(*reverse_it);
                under_test->Insert(ngram);
            }
        }

        under_test->ComputeCountsAndSums(under_test->GetRoot());
    }

    CountTrie *under_test;
};

TEST_F(CountTrieTest, Count) {
    EXPECT_EQ(under_test->Count(std::list<std::string>({})), 0);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"blah"})), 0);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"blah", "blah"})), 0);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"blah", "blah", "blah"})), 0);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"the"})), 3);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"cat"})), 1);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"sat"})), 2);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"the", "cat"})), 1);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"cat", "sat"})), 1);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"sat", "on"})), 2);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"the", "cat", "sat"})), 1);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"sat", "on", "the"})), 2);
    EXPECT_EQ(under_test->Count(std::list<std::string>({"on", "the", "floor"})), 1);
}

TEST_F(CountTrieTest, CountFollowing) {
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({})), 7);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"blah"})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"blah", "blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"the"})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"cat"})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"sat"})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"the", "cat"})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<std::string>({"on", "the"})), 2);
}

TEST_F(CountTrieTest, CountPreceding) {
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"blah"})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"blah", "blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"the"})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"."})), 1);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"sat"})), 2);
    EXPECT_EQ(under_test->CountPreceding(std::list<std::string>({"sat", "on"})), 2);
}

TEST_F(CountTrieTest, CountPrecedingAndFollowing) {
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({})), 8);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"blah"})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"blah", "blah", "blah"})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"the"})), 2);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"on"})), 1);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<std::string>({"floor"})), 1);
}

TEST_F(CountTrieTest, SumFollowing) {
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({})), 11);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"blah"})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"blah", "blah"})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"blah", "blah", "blah"})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"the"})), 3);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"cat"})), 1);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"sat"})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"the", "cat"})), 1);
    EXPECT_EQ(under_test->SumFollowing(std::list<std::string>({"on", "the"})), 2);
}

TEST_F(CountTrieTest, VocabSize) {
    EXPECT_EQ(under_test->VocabSize(), 7);
}
