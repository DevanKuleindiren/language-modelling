#include "ngram.h"
#include "tensorflow/core/platform/test.h"
#include <fstream>

void SetUp(NGram *under_test) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the mouse .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();

    under_test->ProcessFile(test_file_name);
}

TEST(NGramTestProb, ProbBigram) {
    NGram *under_test = new NGram(2, 1);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the"})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat"})), 1/2.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"cat", "ate"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 0.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigramWithMinFreq) {
    NGram *under_test = new NGram(3, 2);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTest, PredictTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);
    std::pair<std::string, double> prediction;

    under_test->Predict(std::list<std::string>({"<s>", "the"}), prediction);
    EXPECT_EQ(prediction.first, "cat");
    ASSERT_DOUBLE_EQ(prediction.second, 2/3.0);

    under_test->Predict(std::list<std::string>({"the", "mouse"}), prediction);
    EXPECT_EQ(prediction.first, ".");
    ASSERT_DOUBLE_EQ(prediction.second, 1.0);
}

TEST(NGramTest, PredictTopKTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);
    std::list<std::pair<std::string, double>> predictions;

    std::list<std::pair<std::string, double>> expected_predictions;
    expected_predictions.push_back(std::make_pair("cat", 2/3.0));
    expected_predictions.push_back(std::make_pair("dog", 1/3.0));

    under_test->PredictTopK(std::list<std::string>({"<s>", "the"}), predictions, 2);

    EXPECT_EQ(predictions, expected_predictions);
}
