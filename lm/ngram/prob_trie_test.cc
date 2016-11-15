#include "gtest/gtest.h"
#include "prob_trie.h"


class ProbTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new ProbTrie(3);
    }

    ProbTrie *under_test;
};

TEST_F(ProbTrieTest, EmptyTrie) {
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the", "cat"})), 0);
}

TEST_F(ProbTrieTest, OneWord) {
    under_test->Insert(std::list<std::string>({"cat"}), 0.2, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the", "cat"})), 0.2);
}

TEST_F(ProbTrieTest, TwoWordsPartial) {
    under_test->Insert(std::list<std::string>({"cat"}), 0.2, 0.6);
    under_test->Insert(std::list<std::string>({"the", "cat"}), 0.5, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah", "blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah", "blah", "blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"a", "cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the", "cat"})), 0.7);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"saw", "the", "cat"})), 0.7);
}

TEST_F(ProbTrieTest, TwoWordsFull) {
    under_test->Insert(std::list<std::string>({"cat"}), 0.2, 0.6);
    under_test->Insert(std::list<std::string>({"the"}), 0.9, 0.1);
    under_test->Insert(std::list<std::string>({"the", "cat"}), 0.5, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah", "blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"blah", "blah", "blah"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the"})), 0.9);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"a", "cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the", "cat"})), 0.52);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"the", "dog"})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"saw", "the", "cat"})), 0.52);
}

TEST_F(ProbTrieTest, ThreeWords) {
    under_test->Insert(std::list<std::string>({"cat"}), 0.2, 0.6);
    under_test->Insert(std::list<std::string>({"the"}), 0.7, 0.1);
    under_test->Insert(std::list<std::string>({"the", "cat"}), 0.3, 0.2);
    under_test->Insert(std::list<std::string>({"saw", "the"}), 0.05, 0.9);
    under_test->Insert(std::list<std::string>({"saw", "the", "cat"}), 0.15, 0.5);
    under_test->Insert(std::list<std::string>({"see", "the", "cat"}), 0.45, 0.0);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"saw", "the", "cat"})), 0.438);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"saw", "a", "cat"})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"fed", "the", "cat"})), 0.32);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<std::string>({"see", "the", "cat"})), 0.77);
}
