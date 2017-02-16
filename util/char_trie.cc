#include "char_trie.h"


void CharTrie::Insert(std::string word, double prob) {
    CharTrie::Node *curr = root;

    probs[word] = prob;
    for (std::string::iterator it = word.begin(); it != word.end(); ++it) {
        if (curr->children.empty() || curr->children.count(*it) < 1) {
            curr->children.insert(std::make_pair(*it, new CharTrie::Node(false)));
        }
        curr = curr->children[*it];
    }
    curr->is_word = true;
}

bool CharTrie::Update(std::string word, double prob) {
    if (probs.count(word) > 0) {
        probs[word] = prob;
        return true;
    } else {
        return false;
    }
}

double CharTrie::GetProb(std::string word) {
    if (probs.count(word) > 0) {
        return probs[word];
    }
    return 0;
}

std::pair<std::string, double> CharTrie::GetMaxWithPrefix(std::string prefix) {
    std::list<std::pair<std::string, double>> top = GetMaxKWithPrefix(prefix, 1);
    if (top.size() > 0) {
        return top.front();
    }
    return std::make_pair("", 0);
}

std::list<std::pair<std::string, double>> CharTrie::GetMaxKWithPrefix(std::string prefix, int k) {
    CharTrie::Node *curr = GetNode(prefix);
    std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> min_heap_max_predictions;
    if (curr != NULL) {
        Max(curr, prefix, min_heap_max_predictions, k);
        std::list<std::pair<std::string, double>> top_k;
        while (min_heap_max_predictions.size() > 0) {
            top_k.push_front(min_heap_max_predictions.top());
            min_heap_max_predictions.pop();
        }
        return top_k;
    }
    return std::list<std::pair<std::string, double>>({});
}

bool CharTrie::Contains(std::string word) {
    return GetNode(word) != NULL;
}

CharTrie::Node *CharTrie::GetNode(std::string word) {
    if (word.size() > 0) {
        CharTrie::Node *curr = root;

        for (std::string::iterator it = word.begin(); it != word.end(); ++it) {
            if (curr->children.empty() || curr->children.count(*it) < 1) {
                return NULL;
            }
            curr = curr->children[*it];
        }
        return curr;
    }
    return root;
}

void CharTrie::Max(CharTrie::Node *node, std::string prefix,
    std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> &min_heap_max_predictions,
    int k) {

    double prob = probs[prefix];
    if (prob > 0) {
        if (min_heap_max_predictions.size() < k) {
            min_heap_max_predictions.push(std::make_pair(prefix, prob));
        } else {
            double min_of_max_k = min_heap_max_predictions.top().second;
            if (prob > min_of_max_k) {
                min_heap_max_predictions.pop();
                min_heap_max_predictions.push(std::make_pair(prefix, prob));
            }
        }
    }

    for (std::unordered_map<char, Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
        Max(it->second, prefix + it->first, min_heap_max_predictions, k);
    }
}
