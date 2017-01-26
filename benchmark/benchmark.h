#ifndef benchmark_h
#define benchmark_h

#include <iostream>
#include <fstream>
#include <math.h>
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/util/char_trie.h"
#include "tensorflow/Source/util/reader.h"

class Benchmark {
    LM *language_model;
public:
    Benchmark(LM *language_model) : language_model(language_model) {}
    double Perplexity(std::string, bool);
    double AverageKeysSaved(std::string, int);
};

#endif // benchmark.h
