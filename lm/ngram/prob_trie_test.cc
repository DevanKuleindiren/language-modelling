#include "prob_trie.h"
#include "tensorflow/core/platform/test.h"


class ProbTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new ProbTrie();
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

TEST_F(ProbTrieTest, MoreThanN) {
    // Here we assume the n-gram is a trigram.
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);

    // Giving the leaf nodes a backoff of 1 ensures that sequences of length greater than N are handled correctly.
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 1.0);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 1.0);

    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({5, 4, 1, 2})), 0.438);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({4, 4, 3, 2})), 0.2);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({6, 7, 1, 2})), 0.32);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({2, 9, 6, 1, 2})), 0.77);
    ASSERT_DOUBLE_EQ(under_test->GetProb(std::list<size_t>({2, 1, 1, 6, 1, 2})), 0.77);
}

TEST_F(ProbTrieTest, EqualsOpsTrue) {
    under_test->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    under_test->Insert(std::list<size_t>({4, 1}), 0.05, 0.9);
    under_test->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ProbTrie *under_test_clone = new ProbTrie();
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

    ProbTrie *under_test_false_clone = new ProbTrie();
    under_test_false_clone->Insert(std::list<size_t>({2}), 0.2, 0.6);
    under_test_false_clone->Insert(std::list<size_t>({1}), 0.7, 0.1);
    under_test_false_clone->Insert(std::list<size_t>({1, 2}), 0.3, 0.2);
    // This line inserts a node with a slightly different backoff.
    under_test_false_clone->Insert(std::list<size_t>({4, 1}), 0.05, 0.4);
    under_test_false_clone->Insert(std::list<size_t>({4, 1, 2}), 0.15, 0.5);
    under_test_false_clone->Insert(std::list<size_t>({6, 1, 2}), 0.45, 0.0);

    ASSERT_FALSE(*under_test == *under_test_false_clone);
}

TEST_F(ProbTrieTest, ToAndFromEqual) {
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

    ProbTrie *under_test_loaded = ProbTrie::FromProto(under_test->ToProto());

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
