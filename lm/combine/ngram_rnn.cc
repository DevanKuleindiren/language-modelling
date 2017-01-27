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
        while (prev_words.size() > ngram_lm->ContextSize().second) {
            prev_words.pop_front();
        }
    } else {
        prev_words = seq;
    }

    std::list<std::pair<std::string, double>> ngram_probs;
    std::list<std::pair<std::string, double>> rnn_probs;

    ngram_lm->ProbAllFollowing(prev_words, ngram_probs);
    rnn_lm->ProbAllFollowing(seq, rnn_probs, use_prev_state);

    std::list<std::pair<std::string, double>>::iterator it_ngram = ngram_probs.begin();
    std::list<std::pair<std::string, double>>::iterator it_rnn = rnn_probs.begin();
    for (; it_ngram != ngram_probs.end() && it_rnn != rnn_probs.end(); ++it_ngram, ++it_rnn) {
        probs.push_back(std::make_pair(it_ngram->first, CombineFunction(it_ngram->second, it_rnn->second)));
    }
}
