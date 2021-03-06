#include "count_trie.h"
#include "tensorflow/core/platform/test.h"


class CountTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new CountTrie(3);
        vocab = new Vocab(1);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/count_trie_test_file";
        test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
        test_file << " the cat sat on the mat . \n";
        test_file << " the cat ate the mouse . \n";
        test_file << " the dog sat on the cat . \n";
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
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({7})), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2, 3})), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({5, 2})), 2);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 4})), 1);
}

TEST_F(CountTrieTest, CountFollowingSpecific) {
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({}), 0, true), 10);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({}), 1, false), 4);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({}), 2, true), 6);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({}), 2, false), 2);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({}), 6, false), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11}), 1, true), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11, 11}), 3, false), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({11, 11, 11}), 1, true), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2}), 1, false), 3);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2}), 2, false), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2}), 2, true), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2}), 3, false), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({2}), 4, false), 0);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 4}), 1, true), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 4}), 1, false), 1);
    EXPECT_EQ(under_test->CountFollowing(std::list<size_t>({10, 4}), 2, false), 0);
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
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({})), 15);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({2})), 5);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->CountPrecedingAndFollowing(std::list<size_t>({8})), 1);
}

TEST_F(CountTrieTest, SumFollowing) {
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({})), 23);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11, 11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({11, 11, 11})), 0);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({2})), 6);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({3})), 3);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({4})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({9})), 1);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({7})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({5, 2})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({4, 5})), 2);
    EXPECT_EQ(under_test->SumFollowing(std::list<size_t>({2, 3})), 3);
}

TEST_F(CountTrieTest, VocabSize) {
    EXPECT_EQ(under_test->VocabSize(), 9);
}

TEST_F(CountTrieTest, CountNGrams) {
    std::vector<std::vector<int>> expected_n_r (4, std::vector<int>(5));
    expected_n_r[1][0] = -1;
    expected_n_r[1][1] = 4;
    expected_n_r[1][2] = 2;
    expected_n_r[1][3] = 3;
    expected_n_r[2][0] = 66;
    expected_n_r[2][1] = 10;
    expected_n_r[2][2] = 3;
    expected_n_r[2][3] = 2;
    expected_n_r[3][0] = 711;
    expected_n_r[3][1] = 15;
    expected_n_r[3][2] = 3;

    std::vector<std::vector<int>> actual_n_r (4, std::vector<int>(5));
    under_test->CountNGrams(&actual_n_r);

    EXPECT_EQ(actual_n_r, expected_n_r);
}
