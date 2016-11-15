#include "kneser_ney.h"

void KneserNey::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<std::string> seq) {
    if (depth > 0) {

        // Calculate pseudo_prob.
        double pseudo_prob = 0;
        double pseudo_prob_numerator = 0;
        double pseudo_prob_denominator = 0;
        if (depth == 1) {
            pseudo_prob_numerator = countTrie->CountPreceding(seq);
            pseudo_prob_denominator = countTrie->CountPrecedingAndFollowing(std::list<std::string>({}));
        } else if (depth == n) {
            pseudo_prob_numerator = std::max(countTrie->Count(seq) - discount, 0.0);
            std::string last_word = seq.back();
            seq.pop_back();
            pseudo_prob_denominator = countTrie->SumFollowing(seq);
            seq.push_back(last_word);
        } else {
            pseudo_prob_numerator = std::max(countTrie->CountPreceding(seq) - discount, 0.0);
            std::string last_word = seq.back();
            seq.pop_back();
            pseudo_prob_denominator = countTrie->CountPrecedingAndFollowing(seq);
            seq.push_back(last_word);
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
            backoff = (discount * countTrie->CountFollowing(seq)) / backoff_denominator;
        }

        prob_trie->Insert(seq, pseudo_prob, backoff);
    }
    for (std::unordered_map<std::string, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
        seq.push_back(it->first);
        PopulateProbTrie(countTrie, it->second, depth + 1, seq);
        seq.pop_back();
    }
}
