#include <iostream>
#include <fstream>
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/Source/lm/ngram/load.h"
#include "tensorflow/Source/lm/rnn/rnn.h"

#define RNN_TYPE std::string("rnn")
#define NGRAM_TYPE std::string("ngram")

void usage(char* const argv_0) {
    std::cerr << "Usage: " << argv_0;
    std::cerr << " --model_path=PATH --type=TYPE --generate=GENR --finish_the_sentence=FNSH --prob_only=PROB" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    PATH is the path the directory containing the model protos." << std::endl;
    std::cerr << "    TYPE is the type of language model (one of: " << RNN_TYPE << " or " << NGRAM_TYPE << ")." << std::endl;
    std::cerr << "    GENR is the number of words to generate using the language model." << std::endl;
    std::cerr << "    FNSH is set if you want the LM to finish your sentence." << std::endl;
    std::cerr << "    PROB is set if you want the LM to print P(w|context) rather than predict the next word." << std::endl;
}

std::list<std::pair<std::string, double>> GetNextK(LM *lm, std::list<std::string> seq, int k) {
    std::list<std::pair<std::string, double>> result;
    for (int i = 0; i < k; i++) {
        std::list<std::pair<std::string, double>> probs;
        if (RNN* rnn = dynamic_cast<RNN*>(lm)) {
            rnn->ProbAllFollowing(seq, probs, false);
        } else {
            lm->ProbAllFollowing(seq, probs);
        }
        double max_prob = 0;
        std::string max_prediction;
        for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
            if (it->second > max_prob && (it->first).compare("<unk>") != 0) {
                max_prob = it->second;
                max_prediction = it->first;
            }
        }
        seq.push_back(max_prediction);
        result.push_back(std::make_pair(max_prediction, max_prob));
    }
    return result;
}

int main(int argc, char* argv[]) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string model_path;
    std::string type;
    int generate = 0;
    bool finish_the_sentence = false;
    bool prob_only = false;

    const bool parse_result = tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("model_path", &model_path),
        tensorflow::Flag("type", &type),
        tensorflow::Flag("generate", &generate),
        tensorflow::Flag("finish_the_sentence", &finish_the_sentence),
        tensorflow::Flag("prob_only", &prob_only),
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
    if (type.compare(RNN_TYPE) == 0) {
        lm = new RNN(model_path);
    } else if (type.compare(NGRAM_TYPE) == 0) {
        lm = Load(model_path);
    } else {
        std::cerr << type << " is not a valid --type." << std::endl;
        return -1;
    }

    if (generate > 0) {
        std::list<std::string> seq;
        seq.push_back("<s>");
        std::list<std::pair<std::string, double>> rest = GetNextK(lm, seq, generate);
        for (std::list<std::pair<std::string, double>>::iterator it = rest.begin(); it != rest.end(); ++it) {
            seq.push_back(it->first);
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

        if (prob_only) {
            std::cout << "Prob = ";
            double prob;
            if (RNN* rnn = dynamic_cast<RNN*>(lm)) {
                prob = rnn->Prob(seq, false);
            } else {
                prob = lm->Prob(seq);
            }
            std::cout << prob << std::endl;
        } else {
            std::cout << "Prediction: ";
            std::list<std::pair<std::string, double>> next = GetNextK(lm, seq, 1);
            if (finish_the_sentence) {
                int count = 0;
                for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
                    std::cout << *it << " ";
                }
                while (next.front().first.compare("<s>") != 0 && count < 30) {
                    seq.push_back(next.front().first);
                    std::cout << next.front().first << " ";
                    next = GetNextK(lm, seq, 1);
                    count++;
                }
                std::cout << "." << std::endl;
            } else {
                std::cout << next.front().first << " (" << next.front().second  << ")"<< std::endl;
            }
        }
        std::cout << std::endl;
    }
}
