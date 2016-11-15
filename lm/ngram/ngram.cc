#include "lm/ngram/ngram.h"

bool NGram::ContainsWord(std::string word) {
    return vocab.count(word) > 0;
}

void NGram::Predict(std::list<std::string> seq, std::pair<std::string, double> &prediction) {
    if (!trained) {
        throw UntrainedException();
    }

    double max_prob = 0;
    std::string max_prediction;
    for (std::unordered_set<std::string>::iterator it = vocab.begin(); it != vocab.end(); ++it) {
        seq.push_back(*it);
        double p = prob_trie->GetProb(seq);
        seq.pop_back();
        if (p > max_prob) {
            max_prob = p;
            max_prediction = (*it);
        }
    }
    prediction = std::make_pair(max_prediction, max_prob);
}

void NGram::PredictTopK(std::list<std::string> seq, std::list<std::pair<std::string, double>> &predictions) {
    if (!trained) {
        throw UntrainedException();
    }

    // TODO
}

double NGram::Prob(std::list<std::string> seq) {
    if (!trained) {
        throw UntrainedException();
    }
    return prob_trie->GetProb(seq);
}

void NGram::ProcessFile(std::string file_name) {
    CountTrie *countTrie = new CountTrie(n);
    countTrie->ProcessFile(file_name);
    countTrie->PopulateVocab(&vocab);

    std::cout << "Populating probability trie..." << std::endl;
    std::list<std::string> seq;
    PopulateProbTrie(countTrie, countTrie->GetRoot(), 0, seq);
    std::cout << "Done." << std::endl;

    trained = true;
}

void NGram::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<std::string> seq) {
    if (depth < n) {
        for (std::unordered_map<std::string, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrie(countTrie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    } else if (depth == n) {

        double count = countTrie->Count(seq);
        std::string last_word = seq.back();
        seq.pop_back();
        double sum_following = countTrie->SumFollowing(seq);
        seq.push_back(last_word);

        prob_trie->Insert(seq, count / sum_following, 0);
    }
}
