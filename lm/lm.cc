#include "lm.h"

bool LM::ContainsWord(std::string word) {
    return vocab->Get(word);
}

void LM::Predict(std::list<std::string> seq, std::pair<std::string, double> &prediction) {
    if (!trained) {
        throw UntrainedException();
    }

    std::list<std::pair<std::string, double>> probs;
    ProbAllFollowing(seq, probs);

    double max_prob = 0;
    std::string max_prediction;
    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        if (it->second > max_prob) {
            max_prob = it->second;
            max_prediction = (it->first);
        }
    }
    prediction = std::make_pair(max_prediction, max_prob);
}

void LM::PredictTopK(std::list<std::string> seq, std::list<std::pair<std::string, double>> &predictions, int k) {
    if (!trained) {
        throw UntrainedException();
    }

    std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> min_heap_max_predictions;
    std::list<std::pair<std::string, double>> probs;
    ProbAllFollowing(seq, probs);

    for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
        if (min_heap_max_predictions.size() < k) {
            min_heap_max_predictions.push(std::make_pair(it->first, it->second));
        } else {
            double min_of_max_k = min_heap_max_predictions.top().second;
            if (it->second > min_of_max_k) {
                min_heap_max_predictions.pop();
                min_heap_max_predictions.push(std::make_pair(it->first, it->second));
            }
        }
    }

    while (min_heap_max_predictions.size() > 0) {
        predictions.push_front(min_heap_max_predictions.top());
        min_heap_max_predictions.pop();
    }
}

std::list<size_t> LM::WordsToIds(std::list<std::string> seq) {
    std::list<size_t> ids;
    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        ids.push_back(vocab->Get(*it));
    }
    return ids;
}

std::list<size_t> LM::Trim(std::list<size_t> seq, int max) {
    if (seq.size() > max) {
        std::list<size_t> trimmed;
        for (int i = 0; i < max; i++) {
            trimmed.push_front(seq.back());
            seq.pop_back();
        }
        return trimmed;
    }
    return seq;
}
