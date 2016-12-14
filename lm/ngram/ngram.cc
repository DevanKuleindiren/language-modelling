#include "ngram.h"

bool NGram::ContainsWord(std::string word) {
    return vocab->Get(word);
}

void NGram::Predict(std::list<std::string> seq, std::pair<std::string, double> &prediction) {
    if (!trained) {
        throw UntrainedException();
    }

    double max_prob = 0;
    std::string max_prediction;
    std::list<size_t> seq_indices = WordsToIndices(seq);
    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        seq_indices.push_back(it->second);
        double p = prob_trie->GetProb(seq_indices);
        seq_indices.pop_back();
        if (p > max_prob) {
            max_prob = p;
            max_prediction = (it->first);
        }
    }
    prediction = std::make_pair(max_prediction, max_prob);
}

void NGram::PredictTopK(std::list<std::string> seq, std::list<std::pair<std::string, double>> &predictions, int k) {
    if (!trained) {
        throw UntrainedException();
    }

    std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> min_heap_max_predictions;

    std::list<size_t> seq_indices = WordsToIndices(seq);
    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        seq_indices.push_back(it->second);
        double p = prob_trie->GetProb(seq_indices);
        seq_indices.pop_back();

        if (min_heap_max_predictions.size() < k) {
            min_heap_max_predictions.push(std::make_pair(it->first, p));
        } else {
            double min_of_max_k = min_heap_max_predictions.top().second;
            if (p > min_of_max_k) {
                min_heap_max_predictions.pop();
                min_heap_max_predictions.push(std::make_pair(it->first, p));
            }
        }
    }

    while (min_heap_max_predictions.size() > 0) {
        predictions.push_front(min_heap_max_predictions.top());
        min_heap_max_predictions.pop();
    }
}

double NGram::Prob(std::list<std::string> seq) {
    if (!trained) {
        throw UntrainedException();
    }
    return prob_trie->GetProb(WordsToIndices(seq));
}

void NGram::ProcessFile(std::string file_name) {
    vocab->ProcessFile(file_name);
    CountTrie *countTrie = new CountTrie(n);
    countTrie->ProcessFile(file_name, vocab);

    std::cout << "Populating probability trie..." << std::endl;
    std::list<size_t> seq;
    PopulateProbTrie(countTrie, countTrie->GetRoot(), 0, seq);
    std::cout << "Done." << std::endl;

    trained = true;
}

void NGram::Save(std::string file_name) {
    prob_trie->Save(file_name);
}

void NGram::Load(std::string file_name) {
    prob_trie->Load(file_name);
}

std::list<size_t> NGram::WordsToIndices(std::list<std::string> seq) {
    std::list<size_t> indices;
    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        indices.push_back(vocab->Get(*it));
    }
    return indices;
}

void NGram::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<size_t> seq) {
    if (depth < n) {
        for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrie(countTrie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    } else if (depth == n) {

        double count = countTrie->Count(seq);
        size_t last_word_index = seq.back();
        seq.pop_back();
        double sum_following = countTrie->SumFollowing(seq);
        seq.push_back(last_word_index);

        prob_trie->Insert(seq, count / sum_following, 0);
    }
}
