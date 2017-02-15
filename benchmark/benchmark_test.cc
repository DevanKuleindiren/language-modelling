#include <math.h>
#include "benchmark.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/Source/lm/ngram/ngram.h"


class ConstantLanguageModelMock : public LM {
    double prob;
public:
    ConstantLanguageModelMock(double prob) : prob(prob) {}
    bool ContainsWord(std::string) {
        return true;
    }
    std::pair<int, int> ContextSize() {
        return std::make_pair(1, 2);
    }
    void Predict(std::list<std::string>, std::pair<std::string, double> &) {}
    void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int) {}
    double Prob (std::list<std::string>) {
        return prob;
    }
    void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &) {
        std::this_thread::sleep_for (std::chrono::seconds(1));
    }
};

void SetUpTestFile(std::string test_file_name) {
    std::ofstream test_file;
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << " the cat sat on the mat . \n";
    test_file << " the cat ate the rat . \n";
    test_file << " the dog sat on the cat . \n";
    test_file.close();
}

TEST(BenchmarkTest, Perplexity) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    ::SetUpTestFile(test_file_name);

    ConstantLanguageModelMock *lm = new ConstantLanguageModelMock(0.25);
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_DOUBLE_EQ(under_test->Perplexity(test_file_name, false), 4.0);
}

TEST(BenchmarkTest, PerplexityExp) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    ::SetUpTestFile(test_file_name);

    ConstantLanguageModelMock *lm = new ConstantLanguageModelMock(0.25);
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_DOUBLE_EQ(under_test->Perplexity(test_file_name, true), 4.0);
}

TEST(BenchmarkTest, PerplexityZero) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    ::SetUpTestFile(test_file_name);

    ConstantLanguageModelMock *lm = new ConstantLanguageModelMock(0);
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_NEAR(under_test->Perplexity(test_file_name, true), 1e9, 1e-4);
}

class LengthLanguageModelMock : public LM {
    std::list<std::string> pseudo_vocab;
public:
    LengthLanguageModelMock(std::list<std::string> pseudo_vocab) : LM(), pseudo_vocab(pseudo_vocab) {}
    bool ContainsWord(std::string) {
        return true;
    }
    std::pair<int, int> ContextSize() {
        return std::make_pair(1, 2);
    }
    void Predict(std::list<std::string>, std::pair<std::string, double> &) {}
    void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int) {}
    double Prob (std::list<std::string> seq) {
        return (seq.back().length() / 10.0);
    }
    void ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
        for (std::list<std::string>::iterator it = pseudo_vocab.begin(); it != pseudo_vocab.end(); ++it) {
            seq.push_back(*it);
            probs.push_back(std::make_pair(*it, Prob(seq)));
            seq.pop_back();
        }
    }
};

TEST(BenchmarkTest, AverageKeysSavedCalculation) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    std::ofstream test_file;
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << " the wolf walked west on the top of the world . \n";
    test_file << " then he saw a wealthy penguin .\n";
    test_file.close();

    LengthLanguageModelMock *lm = new LengthLanguageModelMock(std::list<std::string>({
        "<unk>", "<s>", "the", "wolf", "walked", "west", "on", "top", "of", "world", ".",
        "then", "he", "saw", "a", "wealthy", "penguin"}));
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_DOUBLE_EQ(under_test->AverageKeysSaved(test_file_name, 1000), 46/64.0);
}

TEST(BenchmarkTest, AverageKeysSavedEndToEnd) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    ::SetUpTestFile(test_file_name);

    LM *ngram = new NGram(test_file_name, 2, 1);
    Benchmark *under_test = new Benchmark(ngram);

    ASSERT_DOUBLE_EQ(under_test->AverageKeysSaved(test_file_name, 1000), 57/58.0);
}

TEST(BenchmarkTest, GuessingEntropy) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    std::ofstream test_file;
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << " a wolf walked carefully in fear . \n";
    test_file.close();

    LengthLanguageModelMock *lm = new LengthLanguageModelMock(std::list<std::string>({
        "<unk>", "<s>", "a", "wolf", "walked", "carefully", "in", "fear", "."}));
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_DOUBLE_EQ(under_test->GuessingEntropy(test_file_name, 1000), log2(14336) / 7.0);
}

TEST(BenchmarkTest, MemoryUsage) {
    ConstantLanguageModelMock *lm = new ConstantLanguageModelMock(0.25);
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_GT(under_test->PhysicalMemoryUsageBytes(), 1e3);
    ASSERT_LT(under_test->PhysicalMemoryUsageBytes(), 1e8);
}

TEST(BenchmarkTest, AverageInferenceTime) {
    std::string test_file_name = "/tmp/benchmark_test_file";
    ::SetUpTestFile(test_file_name);

    ConstantLanguageModelMock *lm = new ConstantLanguageModelMock(0.25);
    Benchmark *under_test = new Benchmark(lm);

    ASSERT_NEAR(under_test->AverageInferenceTimeMicroSeconds(test_file_name, 5), 1e6, 1e5);
}
