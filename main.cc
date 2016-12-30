#include <iostream>
#include <fstream>
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/Source/benchmark/benchmark.h"
#include "tensorflow/Source/lm/ngram/load.h"
#include "tensorflow/Source/lm/ngram/smoothing/add_one.h"
#include "tensorflow/Source/lm/ngram/smoothing/kneser_ney.h"
#include "tensorflow/Source/lm/rnn/lstm.h"

#define RNN std::string("rnn")
#define NGRAM std::string("ngram")

void usage(char* const argv_0) {
    std::cerr << "Usage: " << argv_0;
    std::cerr << " --model_path=PATH --type=TYPE --perplexity=PERP --perplexity_exp=PEXP --generate=GENR" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    PATH is the path the directory containing the model protos." << std::endl;
    std::cerr << "    TYPE is the type of language model (one of: " << RNN << " or " << NGRAM << ")." << std::endl;
    std::cerr << "    PERP is the file path of the data to run the perplexity calculation against." << std::endl;
    std::cerr << "    PEXP is the file path of the data to run the exp perplexity calculation against." << std::endl;
    std::cerr << "    GENR is the number of words to generate using the language model." << std::endl;
}

int main(int argc, char* argv[]) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string model_path;
    std::string type;
    std::string perplexity;
    std::string perplexity_exp;
    int generate = 0;

    const bool parse_result = tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("model_path", &model_path),
        tensorflow::Flag("type", &type),
        tensorflow::Flag("perplexity", &perplexity),
        tensorflow::Flag("perplexity_exp", &perplexity_exp),
        tensorflow::Flag("generate", &generate),
    });
    if (!parse_result) {
        usage(argv[0]);
        return -1;
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (type.empty()) {
        std::cerr << "Error: --type must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }

    LM *lm;
    if (type.compare(RNN) == 0) {
        lm = new LSTM(model_path);
    } else {
        lm = Load(model_path);
    }

    if (!perplexity.empty()) {
        Benchmark *benchmark = new Benchmark(lm);
        std::cout << "Perplexity = " << benchmark->Perplexity(perplexity) << std::endl;
    }

    if (!perplexity_exp.empty()) {
        Benchmark *benchmark = new Benchmark(lm);
        std::cout << "Perplexity (exp) = " << benchmark->PerplexityExp(perplexity_exp) << std::endl;
    }

    if (generate > 0) {
        std::list<std::string> seq;
        seq.push_back("<s>");
        for (int i = 0; i < generate; i++) {
            std::pair<std::string, double> prediction;
            lm->Predict(seq, prediction);
            seq.push_back(prediction.first);
        }
        seq.pop_front();

        for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
            std::cout << *it << " ";
        }
        std::cout << std::endl;
    }

    while (true) {
        std::string context;
        std::cout << "Sequence: ";
        std::string str;
        getline(std::cin, context);

        std::list<std::string> seq;
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
        lm->Predict(seq, prediction);

        std::cout << "Prediction: " << prediction.first << " (" << prediction.second << ")" << std::endl;
    }
}
