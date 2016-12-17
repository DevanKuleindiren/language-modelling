#include "prob_trie.h"
#include "tensorflow/core/platform/test.h"


class ProbTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new ProbTrie(3);
    }

    ProbTrie *under_test;
};

/*
blah -> 0
the -> 1
cat -> 2
a -> 3
saw -> 4
dog -> 5
see -> 6
fed -> 7
*/

TEST_F(ProbTrieTest, EmptyTrie) {
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1, 2})), 0);
}

TEST_F(ProbTrieTest, OneWord) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1, 2})), 0.2);
}

TEST_F(ProbTrieTest, TwoWordsPartial) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1, 2}), 0.5, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0, 0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0, 0, 0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({3, 2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1, 2})), 0.7);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 1, 2})), 0.7);
}

TEST_F(ProbTrieTest, TwoWordsFull) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.9, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.5, 0.8);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0, 0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({0, 0, 0})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1})), 0.9);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({3, 2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1, 2})), 0.52);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({1, 5})), 0);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 1, 2})), 0.52);
}

TEST_F(ProbTrieTest, ThreeWords) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 1, 2})), 0.438);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 3, 2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({7, 1, 2})), 0.32);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({6, 1, 2})), 0.77);
}

// FAILS
TEST_F(ProbTrieTest, EqualsOpsTrue) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ProbTrie *under_test_clone = new ProbTrie(3);
    under_test_clone->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test_clone->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test_clone->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test_clone->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test_clone->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test_clone->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_TRUE(*under_test == *under_test_clone);
}

TEST_F(ProbTrieTest, EqualsOpFalse) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ProbTrie *under_test_false_clone = new ProbTrie(3);
    under_test_false_clone->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test_false_clone->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test_false_clone->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    // This line inserts a node with a slightly different backoff.
    under_test_false_clone->Insert(std::list<size_t>({4, 1}), 0.05, 0.4);
    under_test_false_clone->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test_false_clone->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_FALSE(*under_test == *under_test_false_clone);
}

TEST_F(ProbTrieTest, EqualsOpFalseN) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    // The value of N is not the same
    ProbTrie *under_test_false_clone = new ProbTrie(4);
    under_test_false_clone->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test_false_clone->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test_false_clone->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test_false_clone->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test_false_clone->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test_false_clone->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_FALSE(*under_test == *under_test_false_clone);
}

// FAILS
TEST_F(ProbTrieTest, SaveAndLoadEqual) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 1, 2})), 0.438);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 3, 2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({7, 1, 2})), 0.32);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({6, 1, 2})), 0.77);

    std::string test_file_name = "/tmp/prob_trie_test_file.pbtxt";

    under_test->Save(test_file_name);
    ProbTrie *under_test_loaded = new ProbTrie(3);
    under_test_loaded->Load(test_file_name);

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
