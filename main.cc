#include <iostream>
#include <fstream>
#include "lm/ngram/smoothing/kneser_ney.h"
#include "benchmark/benchmark.h"

int main() {
    int n = 3;
    KneserNey *ngramLM = new KneserNey(n, 0.5);

    ngramLM->ProcessFile("/Users/devankuleindiren/Documents/Work/University/Part_II/Project/Datasets/1 Billion Word/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100");

    while (true) {
        std::string context;
        std::cout << "Sequence: ";
        std::string str;
        getline(std::cin, context);
        if (context.compare("PERPLEXITY") == 0) {
            Benchmark *benchmark = new Benchmark(ngramLM);
            std::cout << "Perplexity = " << benchmark->Perplexity("/Users/devankuleindiren/Documents/Work/University/Part_II/Project/Datasets/1 Billion Word/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050") << std::endl;
        } else {
            std::list<std::string> seq;
            for (int i = 0; i < n - 1; i++) {
                seq.push_back("<s>");
            }
            int pos = 0;
            while (!context.empty()) {
                pos = context.find(" ");
                if (pos == std::string::npos) {
                    pos = context.size();
                }
                seq.push_back(context.substr(0, pos));
                context.erase(0, pos + 1);
            }

            std::pair<std::string, double> prediction;
            ngramLM->Predict(seq, prediction);

            std::cout << "Prediction: " << prediction.first << " (" << prediction.second << ")" << std::endl;
        }
    }
}
