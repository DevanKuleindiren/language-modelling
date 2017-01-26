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
        double prob;
        std::unordered_map<char, Node*> children;
        Node(double prob) : prob(prob), children() {}
    };
protected:
    Node *root;
    Node *GetNode(std::string);
    void Max(CharTrie::Node *,
        std::string,
        std::priority_queue<std::pair<std::string, double>, std::vector<std::pair<std::string, double>>, PredictionCompare> &,
        int k);
public:
    CharTrie() : root(new Node(0)) {}
    void Insert(std::string, double);
    std::pair<std::string, double> GetMaxWithPrefix(std::string);
    std::list<std::pair<std::string, double>> GetMaxKWithPrefix(std::string prefix, int k);
    virtual bool Contains(std::string);
};

#endif // char_trie.h
