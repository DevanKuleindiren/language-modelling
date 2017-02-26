#include "error_rnn.h"


double ErrorCorrectingRNN::Prob(std::list<std::string> seq) {
    return Prob(seq, true);
}

double ErrorCorrectingRNN::Prob(std::list<std::string> seq, bool use_prev_state) {
    size_t next_word_id = vocab->Get(seq.back());
    seq.pop_back();

    std::vector<tensorflow::Tensor> outputs;
    RunCorrections(seq, outputs, predictions_tensor_name, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    return predictions(0, next_word_id);
}

void ErrorCorrectingRNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
    ProbAllFollowing(seq, probs, true);
}

void ErrorCorrectingRNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs, bool use_prev_state) {
    std::vector<tensorflow::Tensor> outputs;
    RunCorrections(seq, outputs, predictions_tensor_name, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        probs.push_back(std::make_pair(it->first, predictions(0, it->second)));
    }
}

void ErrorCorrectingRNN::ProbAllFollowing (std::list<std::string> seq, CharTrie *char_trie) {
    ProbAllFollowing(seq, char_trie, true);
}

void ErrorCorrectingRNN::ProbAllFollowing (std::list<std::string> seq, CharTrie *char_trie, bool use_prev_state) {
    std::vector<tensorflow::Tensor> outputs;
    RunCorrections(seq, outputs, predictions_tensor_name, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        char_trie->Update(it->first, predictions(0, it->second));
    }
}

void ErrorCorrectingRNN::LogitsAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &logits) {
    LogitsAllFollowing(seq, logits, true);
}

void ErrorCorrectingRNN::LogitsAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &logits, bool use_prev_state) {
    std::vector<tensorflow::Tensor> outputs;
    RunCorrections(seq, outputs, logits_tensor_name, use_prev_state);
    auto logits_tensor = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        logits.push_back(std::make_pair(it->first, logits_tensor(0, it->second)));
    }
}

void ErrorCorrectingRNN::LogitsAllFollowing (std::list<std::string> seq, CharTrie *char_trie) {
    LogitsAllFollowing(seq, char_trie, true);
}

void ErrorCorrectingRNN::LogitsAllFollowing (std::list<std::string> seq, CharTrie *char_trie, bool use_prev_state) {
    std::vector<tensorflow::Tensor> outputs;
    RunCorrections(seq, outputs, logits_tensor_name, use_prev_state);
    auto logits_tensor = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        char_trie->Update(it->first, logits_tensor(0, it->second));
    }
}

int ErrorCorrectingRNN::EditDistance(std::string a, std::string b) {
    std::vector<std::vector<int>> distances = std::vector<std::vector<int>>(a.size() + 1, std::vector<int>(b.size() + 1));

    for (int i = 1; i <= a.size(); i++) distances[i][0] = i;
    for (int i = 1; i <= b.size(); i++) distances[0][i] = i;

    for (int i = 1; i <= a.size(); i++) {
        for (int j = 1; j <= b.size(); j++) {
            int cost = 1;
            if (a.at(i - 1) == b.at(j - 1)) {
                cost = 0;
            }

            distances[i][j] = std::min(distances[i - 1][j] + 1,
                                       std::min(distances[i][j - 1] + 1, distances[i - 1][j - 1] + cost));
        }
    }

    return distances[a.size()][b.size()];
}

void ErrorCorrectingRNN::RunCorrection(std::string word, std::vector<tensorflow::Tensor> &outputs, std::string output_tensor_name, bool use_prev_state) {
    size_t correct_word_id = vocab->Get(word);
    if (!vocab->ContainsWord(word) && previous_predictions.NumElements() > 0 && use_prev_state) {
        // Find replacement.
        auto previous_predictions_tensor = previous_predictions.tensor<float, 2>();

        std::list<std::pair<std::string, double>> sorted_predictions;
        for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
            sorted_predictions.push_back(std::make_pair(it->first, previous_predictions_tensor(0, it->second)));
        }
        sorted_predictions.sort(PredictionCompare::Compare);

        for (std::list<std::pair<std::string, double>>::iterator pred_it = sorted_predictions.begin(); pred_it != sorted_predictions.end(); ++pred_it) {
            if (EditDistance(word, pred_it->first) <= 2) {
                correct_word_id = vocab->Get(pred_it->first);
                break;
            }
        }
    }
    RunInference(correct_word_id, outputs, output_tensor_name, use_prev_state);
    previous_predictions = outputs[0];
}

void ErrorCorrectingRNN::RunCorrections(std::list<std::string> seq, std::vector<tensorflow::Tensor> &outputs, std::string output_tensor_name, bool use_prev_state) {
    assert(seq.size() > 0);
    RunCorrection(seq.front(), outputs, output_tensor_name, use_prev_state);
    seq.pop_front();
    for (std::list<std::string>::iterator seq_it = seq.begin(); seq_it != seq.end(); ++seq_it) {
        RunCorrection(*seq_it, outputs, output_tensor_name, true);
    }
}
