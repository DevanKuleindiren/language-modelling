#include "add_one.h"

double AddOne::Prob(std::list<std::string> seq) {
    if (!trained) {
        throw UntrainedException();
    }

    // Add one smoothing doesn't exhibit any backoff.
    if (seq.size() < n) {
        return 0;
    }

    // Trim off and words in the sequence beyond the value of n.
    if (seq.size() > n) {
        std::list<std::string> tmp;
        for (int i = 0; i < n; i++) {
            tmp.push_front(seq.back());
            seq.pop_back();
        }
    }

    std::list<size_t> seq_ids = WordsToIndices(seq);
    std::pair<double, double> values = prob_trie->GetValues(seq_ids);
    double count = values.first;
    seq_ids.pop_back();
    values = prob_trie->GetValues(seq_ids);
    double sum_following = values.second;

    return (count + 1) / (sum_following + vocab->Size());
}

void AddOne::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<size_t> seq) {
    if (depth < n) {
        if (depth == n - 1) {
            prob_trie->Insert(seq, 0, countTrie->SumFollowing(seq));
        }

        for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrie(countTrie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    } else if (depth == n) {
        prob_trie->Insert(seq, countTrie->Count(seq), 0);
    }
}

tensorflow::Source::lm::ngram::NGramProto *AddOne::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = NGram::ToProto();
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::ADD_ONE);
    return ngram_proto;
}
