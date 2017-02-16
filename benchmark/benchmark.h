#ifndef benchmark_h
#define benchmark_h

#include <chrono>
#include <iostream>
#include <fstream>
#include <mach/mach.h>
#include <math.h>
#include <sys/resource.h>
#include <thread>
#include <time.h>
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/rnn/rnn.h"
#include "tensorflow/Source/util/char_trie.h"
#include "tensorflow/Source/util/dual_reader.h"

class Benchmark {
    LM *language_model;
    DualFileReader *PrepareDualReader(std::string, std::string);
public:
    Benchmark(LM *language_model) : language_model(language_model) {}
    double Perplexity(std::string, std::string, bool);
    double AverageKeysSaved(std::string, std::string, int);
    double GuessingEntropy(std::string, std::string, int);
    long PhysicalMemoryUsageBytes();
    long AverageInferenceTimeMicroSeconds(std::string, int);
};

#endif // benchmark.h
