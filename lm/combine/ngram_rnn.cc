#include "ngram_rnn.h"


bool NGramRNN::ContainsWord(std::string word) {
    return ngram_lm->ContainsWord(word);
}

std::pair<int, int> NGramRNN::ContextSize() {
    return std::make_pair(1, 2);
}

double NGramRNN::Prob(std::list<std::string> seq) {
    return Prob(seq, true);
}

double NGramRNN::Prob(std::list<std::string> seq, bool use_prev_state) {
    if (use_prev_state) {
        if (!prev_words.empty()) {
            prev_words.pop_back();
        }
        prev_words.insert(prev_words.end(), seq.begin(), seq.end());
        while (prev_words.size() > ngram_lm->ContextSize().second) {
            prev_words.pop_front();
        }
    } else {
        prev_words = seq;
    }
    return CombineFunction(ngram_lm->Prob(prev_words), rnn_lm->Prob(seq, use_prev_state));
}

void NGramRNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
    ProbAllFollowing(seq, probs, true);
}

void NGramRNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs, bool use_prev_state) {
    if (use_prev_state) {
        prev_words.insert(prev_words.end(), seq.begin(), seq.end());
        while (prev_words.size() >= ngram_lm->ContextSize().second) {
            prev_words.pop_front();
        }
    } else {
        prev_words = seq;
    }

    std::list<std::pair<std::string, double>> ngram_probs;
    std::list<std::pair<std::string, double>> rnn_probs;

    ngram_lm->ProbAllFollowing(prev_words, ngram_probs);
    rnn_lm->ProbAllFollowing(seq, rnn_probs, use_prev_state);

    std::unordered_map<std::string, double> ngram_probs_map;
    for (std::list<std::pair<std::string, double>>::iterator it = ngram_probs.begin(); it != ngram_probs.end(); ++it) {
        ngram_probs_map[it->first] = it->second;
    }
    for (std::list<std::pair<std::string, double>>::iterator it = rnn_probs.begin(); it != rnn_probs.end(); ++it) {
        probs.push_back(std::make_pair(it->first, CombineFunction(ngram_probs_map[it->first], it->second)));
    }
}

void NGramRNN::ProbAllFollowing (std::list<std::string> seq, CharTrie *char_trie) {
    ProbAllFollowing(seq, char_trie, true);
}

void NGramRNN::ProbAllFollowing (std::list<std::string> seq, CharTrie *char_trie, bool use_prev_state) {
    std::list<std::pair<std::string, double>> probs;
    ProbAllFollowing(seq, probs, use_prev_state);
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        char_trie->Update(it->first, it->second);
    }
}
