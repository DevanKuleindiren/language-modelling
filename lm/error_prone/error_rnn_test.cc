#include <list>
#include <unordered_set>
#include "error_rnn.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/Source/util/char_trie.h"

#define ABS_ERROR 1e-6

class ErrorCorrectingRNNTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new ErrorCorrectingRNN("tensorflow/Source/lm/rnn/test_data");
    }

    ErrorCorrectingRNN *under_test;
};

TEST_F(ErrorCorrectingRNNTest, EditDistance) {
    ASSERT_EQ(under_test->EditDistance("the", "the"), 0);
    ASSERT_EQ(under_test->EditDistance("flamingo", "floaango"), 3);
    ASSERT_EQ(under_test->EditDistance("toaster", "toasssteeer"), 4);
    ASSERT_EQ(under_test->EditDistance("watermelon", "wtrmln"), 4);
    ASSERT_EQ(under_test->EditDistance("unicorn", "banana"), 7);
}

// Note: These tests use a dummy LSTM graph protocol buffer, which doesn't actually produce probabilities between 0
// and 1.
TEST_F(ErrorCorrectingRNNTest, ProbWithMemory) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<s>", "the"}), true), 0.6, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>"}), true), 3.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<unk>", "the", "<unk>", "<s>", "the"}), true), 8.1, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>", "<s>"}), true), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbWithoutMemory) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<s>", "the"}), false), 0.6, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>"}), false), 1.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<unk>", "the", "<unk>", "<s>", "the"}), false), 3.9, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>", "<s>"}), false), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbWithMemoryAndCorrectionWithinEditDist2) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<s>", "the"}), true), 0.6, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"h", "<unk>"}), true), 3.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<unk>", "their", "dinasour", "<s>", "the"}), true), 8.1, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"ie", "<unk>", "<s>"}), true), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbWithMemoryAndCorrectionNotWithinEditDist2) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<rrr>", "the"}), true), 0, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"themes", "<unk>"}), true), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbWithoutMemoryAndCorrectionWithinEditDist2) {
    // This first example is within edit distance 2, but because the previous_predictions tensor won't have been
    // initialised yet, correction of <ss> cannot be made, so it is mapped to <unk>.
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<se", "the"}), false), 0, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>"}), false), 1.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<usk>", "thhi", "<uuu>", "<sss>", "the"}), false), 3.9, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<ink>", "<s>"}), false), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbWithoutMemoryAndCorrectionNotWithinEditDist2) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<ss>", "the"}), false), 0, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"blah", "the"}), false), 0, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"tthhee", "<unk>"}), false), 0, ABS_ERROR);
}

TEST_F(ErrorCorrectingRNNTest, ProbAllFollowingWithMemory) {
    std::list<std::pair<std::string, double>> probs;
    std::unordered_map<std::string, double> expected_probs;

    under_test->ProbAllFollowing(std::list<std::string>({"<s>"}), probs, true);
    expected_probs["<unk>"] = 0.4;
    expected_probs["<s>"] = 1.0;
    expected_probs["the"] = 0.6;
    ASSERT_EQ(probs.size(), expected_probs.size());
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        ASSERT_NEAR(expected_probs[it->first], it->second, ABS_ERROR);
    }
    probs.clear();
    expected_probs.clear();

    under_test->ProbAllFollowing(std::list<std::string>({"the", "the"}), probs, true);
    expected_probs["<unk>"] = 6.8;
    expected_probs["<s>"] = 17.0;
    expected_probs["the"] = 10.2;
    ASSERT_EQ(probs.size(), expected_probs.size());
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        ASSERT_NEAR(expected_probs[it->first], it->second, ABS_ERROR);
    }
}

TEST_F(ErrorCorrectingRNNTest, ProbAllFollowingWithoutMemory) {
    std::list<std::pair<std::string, double>> probs;
    std::unordered_map<std::string, double> expected_probs;

    under_test->ProbAllFollowing(std::list<std::string>({"<s>"}), probs, false);
    expected_probs["<unk>"] = 0.4;
    expected_probs["<s>"] = 1.0;
    expected_probs["the"] = 0.6;
    ASSERT_EQ(probs.size(), expected_probs.size());
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        ASSERT_NEAR(expected_probs[it->first], it->second, ABS_ERROR);
    }
    probs.clear();
    expected_probs.clear();

    under_test->ProbAllFollowing(std::list<std::string>({"the", "the"}), probs, false);
    expected_probs["<unk>"] = 4.8;
    expected_probs["<s>"] = 12.0;
    expected_probs["the"] = 7.2;
    ASSERT_EQ(probs.size(), expected_probs.size());
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        ASSERT_NEAR(expected_probs[it->first], it->second, ABS_ERROR);
    }
}

TEST_F(ErrorCorrectingRNNTest, ProbAllFollowingCharTrie) {
    CharTrie *char_trie = new CharTrie();
    char_trie->Insert("<unk>", 0);
    char_trie->Insert("<s>", 0);
    char_trie->Insert("the", 0);

    std::list<std::pair<std::string, double>> actual_top_3;
    under_test->ProbAllFollowing(std::list<std::string>({"<s>"}), char_trie, true);
    actual_top_3 = char_trie->GetMaxKWithPrefix("", 3);
    ASSERT_EQ(actual_top_3.size(), 3);
    ASSERT_STREQ(actual_top_3.front().first.c_str(), "<s>");
    ASSERT_NEAR(actual_top_3.front().second, 1.0, ABS_ERROR);
    actual_top_3.pop_front();
    ASSERT_STREQ(actual_top_3.front().first.c_str(), "the");
    ASSERT_NEAR(actual_top_3.front().second, 0.6, ABS_ERROR);
    actual_top_3.pop_front();
    ASSERT_STREQ(actual_top_3.front().first.c_str(), "<unk>");
    ASSERT_NEAR(actual_top_3.front().second, 0.4, ABS_ERROR);
}
