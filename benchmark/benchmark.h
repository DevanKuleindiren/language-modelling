#include "tensorflow/Source/lm/lm.h"
#include <iostream>
#include <fstream>
#include <math.h>

class Benchmark {
    LM *language_model;
public:
    Benchmark(LM *language_model) : language_model(language_model) {}
    double Perplexity(std::string file_name);
};
