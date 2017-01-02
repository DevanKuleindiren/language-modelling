#include "katz.h"

double Katz::Prob(std::list<std::string> seq) {
    return Prob(WordsToIndices(seq));
}

double Katz::Prob(std::list<size_t> seq) {
    if (!trained) {
        throw UntrainedException();
    }

    // Trim off any words in the sequence beyond the value of n.
    seq = Trim(seq, n);

    std::pair<double, double> values = prob_trie->GetValues(seq);
    if (values.first > 0 || seq.size() <= 1) {
        return values.first;
    } else {
        size_t last_word_index = seq.back();
        seq.pop_back();
        double alpha = (prob_trie->GetValues(seq)).second;
        seq.push_back(last_word_index);
        seq.pop_front();
        return alpha * Prob(seq);
    }
}

void Katz::ProcessFile(std::string file_name) {
    vocab->ProcessFile(file_name);
    CountTrie *count_trie = new CountTrie(Katz::k + 1);
    count_trie->ProcessFile(file_name, vocab);

    std::cout << "Processing probability trie..." << std::endl;

    // Count the number of n-grams for n in the range [1, k + 1].
    std::array<int, (Katz::k + 2)> num_ngrams;
    num_ngrams.fill(0);
    CountNGrams(count_trie->GetRoot(), 0, &num_ngrams);

    // Use these counts to calculate d_i (from the Katz smoothing equation) for i in the range [1, k].
    // (x = ((k + 1)n_{k + 1})/n_1).
    std::array<double, (Katz::k + 1)> adjusted_counts;
    adjusted_counts.fill(0);
    double x = ((Katz::k + 1) * num_ngrams[Katz::k + 1]) / ((double) num_ngrams[1]);
    for (int r = 1; r < num_ngrams.size(); ++r) {
        double r_star = (r + 1) * (num_ngrams[r + 1] / (double) num_ngrams[r]);
        adjusted_counts[r] = (r_star - (r * x)) / (1 - x);
    }

    // Compute P_{katz}(w_i | w_{i - n + 1}^{i - 1}) for all w_{i - n + 1}^i and store them in pseudo_probs.
    std::list<size_t> seq;
    PopulateProbTriePseudoProb(count_trie, count_trie->GetRoot(), &adjusted_counts, 0, seq);

    // Compute a(w_{i - n + 1}^{i - 1}) and store them in backoffs.
    PopulateProbTrieBackoff(prob_trie, prob_trie->GetRoot(), 0, seq);
}

tensorflow::Source::lm::ngram::NGramProto *Katz::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = NGram::ToProto();
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KATZ);
    return ngram_proto;
}

void Katz::CountNGrams(CountTrie::Node *node, int depth, std::array<int, (Katz::k + 2)> *num_ngrams) {
    if (depth < (*num_ngrams).size()) {
        (*num_ngrams)[depth]++;
        for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            CountNGrams(it->second, depth + 1, num_ngrams);
        }
    }
}

void Katz::PopulateProbTriePseudoProb(CountTrie *countTrie,
                                      CountTrie::Node *node,
                                      std::array<double, (Katz::k + 1)> *adjusted_count,
                                      int depth,
                                      std::list<size_t> seq) {
    if (depth > 0) {
        // Calculate pseudo_prob.
        double pseudo_prob = 0;
        double pseudo_prob_numerator = 0;
        double pseudo_prob_denominator = 0;

        int count = countTrie->Count(seq);
        if (count <= Katz::k) {
            pseudo_prob_numerator = (*adjusted_count)[count];
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

        prob_trie->Insert(seq, pseudo_prob, 0);
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
