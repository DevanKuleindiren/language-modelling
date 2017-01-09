#include "katz.h"

Katz::Katz(std::string file_name, int n, int min_frequency) : NGram(n, min_frequency) {
    ProcessFile(file_name);
}

std::pair<int, int> Katz::ContextSize() {
    return std::make_pair(1, n);
}

double Katz::Prob(std::list<std::string> seq) {
    return Prob(WordsToIds(seq));
}

double Katz::Prob(std::list<size_t> seq) {
    // Trim off any words in the sequence beyond the value of n.
    seq = Trim(seq, n);

    if (prob_trie->Contains(seq) || seq.size() <= 1) {
        return (prob_trie->GetValues(seq)).first;
    } else {
        size_t last_word_index = seq.back();
        seq.pop_back();
        while (!prob_trie->Contains(seq) && !seq.empty()) {
            seq.pop_front();
        }
        double alpha = 1;
        if (!seq.empty()) {
            alpha = (prob_trie->GetValues(seq)).second;
            seq.pop_front();
        }
        seq.push_back(last_word_index);
        return alpha * Prob(seq);
    }
}

tensorflow::Source::lm::ngram::NGramProto *Katz::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = NGram::ToProto();
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KATZ);
    return ngram_proto;
}

void Katz::ProcessCountTrie(CountTrie *count_trie) {
    // Compute n_r for n in [1, n] and r in [0, k + 1], where n_r is the number of n-grams that occur r times.
    std::vector<std::vector<int>> n_r (n + 1, std::vector<int>(Katz::k + 2));
    count_trie->CountNGrams(&n_r);

    // Use these counts to calculate d_r (from the Katz smoothing equation) for i in the range [1, k].
    std::vector<std::vector<double>> adjusted_counts (n + 1, std::vector<double>(Katz::k + 1));
    for (int i = 1; i <= n; ++i) {
        if (n_r[i][1] > 0) {
            double x = ((Katz::k + 1) * n_r[i][Katz::k + 1]) / ((double) n_r[i][1]);
            for (int r = 0; r <= Katz::k; ++r) {
                if (n_r[i][r] > 0) {
                    double r_star = (r + 1) * (n_r[i][r + 1] / (double) n_r[i][r]);
                    adjusted_counts[i][r] = (r_star - (r * x)) / (1 - x);
                }
            }
        }
    }

    // Compute P_{katz}(w_i | w_{i - n + 1}^{i - 1}) for all w_{i - n + 1}^i and store them in pseudo_probs.
    std::list<size_t> seq;
    PopulateProbTriePseudoProb(count_trie, count_trie->GetRoot(), &adjusted_counts, 0, seq);

    // Compute a(w_{i - n + 1}^{i - 1}) and store them in backoffs.
    PopulateProbTrieBackoff(prob_trie, prob_trie->GetRoot(), 0, seq);
}

void Katz::PopulateProbTriePseudoProb(CountTrie *countTrie,
                                      CountTrie::Node *node,
                                      std::vector<std::vector<double>> *adjusted_count,
                                      int depth,
                                      std::list<size_t> seq) {
    if (depth > 0) {
        // Calculate pseudo_prob.
        double pseudo_prob = 0;
        double pseudo_prob_numerator = 0;
        double pseudo_prob_denominator = 0;

        int count = countTrie->Count(seq);
        if (count <= Katz::k) {
            pseudo_prob_numerator = (*adjusted_count)[depth][count];
        } else {
            pseudo_prob_numerator = count;
        }

        size_t last_word_index = seq.back();
        seq.pop_back();
        pseudo_prob_denominator = countTrie->SumFollowing(seq);
        seq.push_back(last_word_index);

        if (pseudo_prob_denominator > 0) {
            pseudo_prob = pseudo_prob_numerator / pseudo_prob_denominator;
        }

        prob_trie->Insert(seq, pseudo_prob, 1);
    }

    if (depth < n) {
        for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTriePseudoProb(countTrie, it->second, adjusted_count, depth + 1, seq);
            seq.pop_back();
        }
    }
}

void Katz::PopulateProbTrieBackoff(ProbTrie *prob_trie, ProbTrie::Node *node, int depth, std::list<size_t> seq) {
    if (depth > 0) {

        // Calculate pseudo_prob.
        double alpha = 1;
        double alpha_numerator = 1;
        double alpha_denominator = 1;

        for (std::unordered_map<size_t, ProbTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            alpha_numerator -= (it->second)->pseudo_prob;
        }

        for (std::unordered_map<size_t, ProbTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            size_t first_word_index = seq.front();
            seq.pop_front();
            seq.push_back(it->first);
            alpha_denominator -= (prob_trie->GetValues(seq)).first;
            seq.pop_back();
            seq.push_front(first_word_index);
        }

        if (alpha_denominator != 0) {
            alpha = alpha_numerator / alpha_denominator;
        }

        prob_trie->Insert(seq, node->pseudo_prob, alpha);
    }

    if (depth < (n - 1)) {
        for (std::unordered_map<size_t, ProbTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrieBackoff(prob_trie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    }
}
