#include <list>
#include <unordered_set>
#include "rnn.h"
#include "tensorflow/core/platform/test.h"

#define ABS_ERROR 1e-6

class RNNTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new RNN("tensorflow/Source/lm/rnn/test_data");
    }

    RNN *under_test;
};

// Note: These tests use a dummy LSTM graph protocol buffer, which doesn't actually produce probabilities between 0
// and 1.
TEST_F(RNNTest, ProbWithMemory) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<s>", "the"}), true), 0.6, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>"}), true), 3.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<unk>", "the", "<unk>", "<s>", "the"}), true), 8.1, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>", "<s>"}), true), 0, ABS_ERROR);
}

TEST_F(RNNTest, ProbWithoutMemory) {
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<s>", "the"}), false), 0.6, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>"}), false), 1.2, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"<unk>", "the", "<unk>", "<s>", "the"}), false), 3.9, ABS_ERROR);
    ASSERT_NEAR(under_test->Prob(std::list<std::string>({"the", "<unk>", "<s>"}), false), 0, ABS_ERROR);
}

TEST_F(RNNTest, ProbAllFollowingWithMemory) {
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

TEST_F(RNNTest, ProbAllFollowingWithoutMemory) {
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
