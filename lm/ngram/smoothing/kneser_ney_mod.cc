#include "kneser_ney_mod.h"

KneserNeyMod::KneserNeyMod(std::string file_name, int n, int min_frequency) : KneserNey(n, min_frequency) {
    ProcessFile(file_name);
}

KneserNeyMod::KneserNeyMod(int n, std::list<double> discount_list, ProbTrie *prob_trie, Vocab *vocab) : KneserNey(n, 0, prob_trie, vocab) {
    discounts = std::vector<std::vector<double>>(n + 1, std::vector<double>(3));
    for (int i = 2; i <= n; i++) {
        for (int k = 0; k < 3; k++) {
            discounts[i][k] = discount_list.front();
            discount_list.pop_front();
        }
    }
}

bool KneserNeyMod::operator==(const NGram &to_compare) {
    if (const KneserNeyMod *to_compare_ads = dynamic_cast<const KneserNeyMod*>(&to_compare)) {
        return NGram::operator==(to_compare) && (discounts == to_compare_ads->discounts);
    }
    return false;
}

tensorflow::Source::lm::ngram::NGramProto *KneserNeyMod::ToProto() {
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = NGram::ToProto();
    ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KNESER_NEY_MOD);
    for (int i = 2; i <= n; i++) {
        for (int k = 0; k < 3; k++) {
            ngram_proto->add_discount(discounts[i][k]);
        }
    }
    return ngram_proto;
}

void KneserNeyMod::ProcessCountTrie(CountTrie *count_trie) {
    std::vector<std::vector<int>> n_r (n + 1, std::vector<int>(5));
    count_trie->CountNGrams(&n_r);

    discounts = std::vector<std::vector<double>>(n + 1, std::vector<double>(3));
    for (int i = 2; i <= n; i++) {
        double d = 1;
        if (n_r[i][1] != 0) {
            d = n_r[i][1] / (double) (n_r[i][1] + (2 * n_r[i][2]));
        }

        discounts[i][0] = 1;
        if (n_r[i][1] != 0) discounts[i][0] -= (2 * d * (n_r[i][2] / (double) n_r[i][1]));

        discounts[i][1] = 2;
        if (n_r[i][2] != 0) discounts[i][1] -= (3 * d * (n_r[i][3] / (double) n_r[i][2]));

        discounts[i][2] = 3;
        if (n_r[i][3] != 0) discounts[i][2] -= (4 * d * (n_r[i][4] / (double) n_r[i][3]));
    }

    std::list<size_t> seq;
    PopulateProbTrie(count_trie, count_trie->GetRoot(), 0, seq);
}

double KneserNeyMod::Discount(int n, int count) {
    if (count == 0) {
        return 0;
    } else if (count == 1 || count == 2) {
        return discounts[n][count - 1];
    } else {
        return discounts[n][2];
    }
}

double KneserNeyMod::BackoffNumerator(CountTrie *count_trie, int depth, std::list<size_t> seq) {
    int n_depth = depth + 1;
    return (discounts[n_depth][0] * count_trie->CountFollowing(seq, 1, false))
         + (discounts[n_depth][1] * count_trie->CountFollowing(seq, 2, false))
         + (discounts[n_depth][2] * count_trie->CountFollowing(seq, 3, true));
}
