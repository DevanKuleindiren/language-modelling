#include <iostream>
#include <fstream>
#include "tensorflow/Source/lm/ngram/smoothing/add_one.h"
#include "tensorflow/Source/lm/rnn/lstm.h"
#include "tensorflow/Source/benchmark/benchmark.h"

int main() {
    int n = 3;
    //AddOne *lm_ = new AddOne(n, 3);
    LSTM *lm_ = new LSTM("/Users/devankuleindiren/Desktop/save", 1);

    //lm_->ProcessFile("/Users/devankuleindiren/Documents/Work/University/Part_II/Project/Datasets/1-Billion-Word/1-billion-word-language-modeling-benchmark-r13output/training-monolingual.tokenized.shuffled/news.en-00001-of-00100");

    while (true) {
        std::string context;
        std::cout << "Sequence: ";
        std::string str;
        getline(std::cin, context);
        if (context.compare("PERPLEXITY") == 0) {
            Benchmark *benchmark = new Benchmark(lm_);
            std::cout << "Perplexity = " << benchmark->Perplexity("/Users/devankuleindiren/Documents/Work/University/Part_II/Project/Datasets/1-Billion-Word/1-billion-word-language-modeling-benchmark-r13output/heldout-monolingual.tokenized.shuffled/news.en.heldout-00000-of-00050") << std::endl;
        } else if (context.compare("GENERATE") == 0) {
            std::cout << "Number of words: ";
            int num_words;
            std::cin >> num_words;

            std::list<std::string> seq;
            for (int i = 0; i < n - 1; i++) {
                seq.push_back("<s>");
            }
            for (int i = 0; i < num_words; i++) {
                std::pair<std::string, double> prediction;
                lm_->Predict(seq, prediction);
                seq.push_back(prediction.first);
            }
            for (int i = 0; i < n - 1; i++) {
                seq.pop_front();
            }
            for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
                std::cout << *it << " ";
            }
            std::cout << std::endl;
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
            lm_->Predict(seq, prediction);

            std::cout << "Prediction: " << prediction.first << " (" << prediction.second << ")" << std::endl;
        }
    }
}
