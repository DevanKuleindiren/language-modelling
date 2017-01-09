#include "add_one.h"

AddOne::AddOne(std::string file_name, int n, int min_frequency) : NGram(n, min_frequency) {
     ProcessFile(file_name);
}
double AddOne::Prob(std::list<std::string> seq) {
    return Prob(WordsToIds(seq));
}

double AddOne::Prob(std::list<size_t> seq) {
    // Add one smoothing doesn't exhibit any backoff.
    if (seq.size() < n) {
        return 0;
    }

    // Trim off any words in the sequence beyond the value of n.
    seq = Trim(seq, n);

    std::pair<double, double> values = prob_trie->GetValues(seq);
    double count = values.first;
    seq.pop_back();
    values = prob_trie->GetValues(seq);
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
