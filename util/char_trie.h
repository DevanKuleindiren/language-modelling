#ifndef char_trie_h
#define char_trie_h

#include <list>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "prediction_compare.h"

class CharTrie {
public:
    struct Node {
        bool is_word;
        std::unordered_map<char, Node*> children;
        Node(bool is_word) : is_word(is_word), children() {}
    };
protected:
    Node *root;
    std::unordered_map<std::string, double> probs;
    Node *GetNode(std::string);
    void Max(CharTrie::Node *,
        std::string,
        std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> &,
        int k);
public:
    CharTrie() : root(new Node(false)) {}
    void Insert(std::string, double);
    bool Update(std::string, double);
    std::pair<std::string, double> GetMaxWithPrefix(std::string);
    std::list<std::pair<std::string, double>> GetMaxKWithPrefix(std::string prefix, int k);
    virtual bool Contains(std::string);
};

#endif // char_trie.h
