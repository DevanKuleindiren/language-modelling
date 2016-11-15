#ifndef prob_trie_h
#define prob_trie_h

#include <iostream>
#include <list>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <utility>

#endif /* prob_trie_h */


class ProbTrie {
protected:
    struct Node {
        double pseudo_prob;
        double backoff;
        std::unordered_map<std::string, Node*> children;
        Node(double pseudo_prob, double backoff) : pseudo_prob(pseudo_prob), backoff(backoff), children() {}
    };
    int n;
    Node *root;
    Node *GetNode(std::list<std::string>);
    double GetProbRecurse(std::list<std::string>);
public:
    ProbTrie(int n) : n(n), root(new Node(0, 0)) {}
    void Insert(std::list<std::string>, double, double);
    virtual double GetProb(std::list<std::string>);
};
