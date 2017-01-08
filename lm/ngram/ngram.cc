#include "ngram.h"

std::pair<int, int> NGram::ContextSize() {
    return std::make_pair(n - 1, n);
}

double NGram::Prob(std::list<std::string> seq) {
    return Prob(WordsToIds(seq));
}

double NGram::Prob(std::list<size_t> seq) {
    if (!trained) {
        throw UntrainedException();
    }

    // Trim off any words in the sequence beyond the value of n.
    seq = Trim(seq, n);

    return prob_trie->GetProb(seq);
}

void NGram::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
    if (!trained) {
        throw UntrainedException();
    }

    std::list<size_t> seq_ids = WordsToIds(seq);
    seq_ids = Trim(seq_ids, n - 1);
    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        seq_ids.push_back(it->second);
        probs.push_back(std::make_pair(it->first, Prob(seq_ids)));
        seq_ids.pop_back();
    }
}

void NGram::ProcessFile(std::string file_name) {
    vocab->ProcessFile(file_name);
    CountTrie *count_trie = new CountTrie(n);
    count_trie->ProcessFile(file_name, vocab);
    std::cout << "Processing probability trie..." << std::endl;
    ProcessCountTrie(count_trie);
    trained = true;
}

bool NGram::operator==(const NGram &to_compare) {
    return (n == to_compare.n)
        && (*prob_trie == *to_compare.prob_trie)
        && (*vocab == *to_compare.vocab)
        && (trained == to_compare.trained);
}

tensorflow::Source::lm::ngram::NGramProto *NGram::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    ngram_proto->set_n(n);
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::NONE);
    ngram_proto->set_allocated_prob_trie(prob_trie->ToProto());
    return ngram_proto;
}

void NGram::Save(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }
    vocab->Save(directory_path + "vocab.pbtxt");

    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = ToProto();
    std::ofstream ofs (directory_path + "ngram.pbtxt", std::ios::out | std::ios::trunc);
    google::protobuf::io::OstreamOutputStream osos(&ofs);
    if (!google::protobuf::TextFormat::Print(*ngram_proto, &osos)) {
        std::cerr << "Failed to write ngram proto." << std::endl;
    } else {
        std::cout << "Saved ngram proto." << std::endl;
    }
}

void NGram::ProcessCountTrie(CountTrie *count_trie) {
    std::list<size_t> seq;
    PopulateProbTrie(count_trie, count_trie->GetRoot(), 0, seq);
}

void NGram::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<size_t> seq) {
    if (depth < n) {
        for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrie(countTrie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    } else if (depth == n) {
        double count = countTrie->Count(seq);
        size_t last_word_index = seq.back();
        seq.pop_back();
        double sum_following = countTrie->SumFollowing(seq);
        seq.push_back(last_word_index);

        prob_trie->Insert(seq, count / sum_following, 0);
    }
}
