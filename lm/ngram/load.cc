#include "load.h"

NGram *Load(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    Vocab *vocab = Vocab::Load(directory_path + "vocab.pbtxt");

    std::ifstream ifs (directory_path + "ngram.pbtxt", std::ios::in);
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();

    google::protobuf::io::IstreamInputStream isis(&ifs);
    if (!google::protobuf::TextFormat::Parse(&isis, ngram_proto)) {
        std::cerr << "Failed to read ngram proto." << std::endl;
    } else {
        std::cout << "Read ngram proto." << std::endl;
    }

    NGram *ngram;
    int n = ngram_proto->n();
    ProbTrie *prob_trie = ProbTrie::FromProto(&(ngram_proto->prob_trie()));

    switch (ngram_proto->smoothing()) {
        case tensorflow::Source::lm::ngram::Smoothing::ADD_ONE: {
            ngram = new AddOne(n, prob_trie, vocab);
            break;
        }
        case tensorflow::Source::lm::ngram::Smoothing::KNESER_NEY: {
            ngram = new KneserNey(n, ngram_proto->discount(), prob_trie, vocab);
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
