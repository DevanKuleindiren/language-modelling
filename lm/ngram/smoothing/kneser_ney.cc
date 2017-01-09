#include "kneser_ney.h"

KneserNey::KneserNey(std::string file_name, int n, int min_frequency) : AbsoluteDiscounting(n, min_frequency) {
    ProcessFile(file_name);
}

tensorflow::Source::lm::ngram::NGramProto *KneserNey::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = NGram::ToProto();
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KNESER_NEY);
    ngram_proto->add_discount(discount);
    return ngram_proto;
}

void KneserNey::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<size_t> seq) {
    if (depth > 0) {

        // Calculate pseudo_prob.
        double pseudo_prob = 0;
        double pseudo_prob_numerator = 0;
        double pseudo_prob_denominator = 0;
        if (depth == 1) {
            pseudo_prob_numerator = countTrie->CountPreceding(seq);
            pseudo_prob_denominator = countTrie->CountPrecedingAndFollowing(std::list<size_t>({}));
        } else if (depth == n) {
            int count = countTrie->Count(seq);
            pseudo_prob_numerator = std::max(count - Discount(depth, count), 0.0);
            size_t last_word_index = seq.back();
            seq.pop_back();
            pseudo_prob_denominator = countTrie->SumFollowing(seq);
            seq.push_back(last_word_index);
        } else {
            pseudo_prob_numerator = std::max(countTrie->CountPreceding(seq) - Discount(depth, countTrie->Count(seq)), 0.0);
            size_t last_word_index = seq.back();
            seq.pop_back();
            pseudo_prob_denominator = countTrie->CountPrecedingAndFollowing(seq);
            seq.push_back(last_word_index);
        }
        if (pseudo_prob_denominator > 0) {
            pseudo_prob = pseudo_prob_numerator / pseudo_prob_denominator;
        }

        // Calculate backoff.
        double backoff = 1;
        double backoff_denominator = 0;
        if (depth == n - 1) {
            backoff_denominator = countTrie->SumFollowing(seq);
        } else {
            backoff_denominator = countTrie->CountPrecedingAndFollowing(seq);
        }
        if (backoff_denominator > 0) {
            backoff = BackoffNumerator(countTrie, depth, seq) / backoff_denominator;
        }

        prob_trie->Insert(seq, pseudo_prob, backoff);
    }
    for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
        seq.push_back(it->first);
        PopulateProbTrie(countTrie, it->second, depth + 1, seq);
        seq.pop_back();
    }
}

double KneserNey::Discount(int n, int count) {
    return discount;
}

double KneserNey::BackoffNumerator(CountTrie *count_trie, int depth, std::list<size_t> seq) {
    return discount * count_trie->CountFollowing(seq);
}
