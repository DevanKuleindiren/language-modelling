#include "load.h"

NGram *Load(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    Vocab *vocab = Vocab::Load(directory_path + "vocab.pbtxt");

    std::fstream ifs (directory_path + "ngram.pb", std::ios::in | std::ios::binary);
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    if (!ngram_proto->ParseFromIstream(&ifs)) {
        std::cerr << "Failed to read ngram proto." << std::endl;
    } else {
        std::cout << "Read ngram proto." << std::endl;
    }

    NGram *ngram;
    int n = ngram_proto->n();
    ProbTrie *prob_trie = ProbTrie::FromProto(&(ngram_proto->prob_trie()));

    switch (ngram_proto->smoothing()) {
        case tensorflow::Source::lm::ngram::Smoothing::ABSOLUTE_DISCOUNTING: {
            ngram = new AbsoluteDiscounting(n, ngram_proto->discount(0), prob_trie, vocab);
            break;
        }
        case tensorflow::Source::lm::ngram::Smoothing::ADD_ONE: {
            ngram = new AddOne(n, prob_trie, vocab);
            break;
        }
        case tensorflow::Source::lm::ngram::Smoothing::KATZ: {
            ngram = new Katz(n, prob_trie, vocab);
            break;
        }
        case tensorflow::Source::lm::ngram::Smoothing::KNESER_NEY: {
            ngram = new KneserNey(n, ngram_proto->discount(0), prob_trie, vocab);
            break;
        }
        case tensorflow::Source::lm::ngram::Smoothing::KNESER_NEY_MOD: {
            std::list<double> discounts;
            for (int i = 0; i < ngram_proto->discount_size(); i++) {
                discounts.push_back(ngram_proto->discount(i));
            }
            ngram = new KneserNeyMod(n, discounts, prob_trie, vocab);
            break;
        }
        default: {
            ngram = new NGram(n, prob_trie, vocab);
            break;
        }
    }

    ifs.close();
    return ngram;
}
