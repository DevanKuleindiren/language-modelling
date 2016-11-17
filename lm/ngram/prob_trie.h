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
        std::unordered_map<size_t, Node*> children;
        Node(double pseudo_prob, double backoff) : pseudo_prob(pseudo_prob), backoff(backoff), children() {}
    };
    int n;
    Node *root;
    Node *GetNode(std::list<size_t>);
public:
    ProbTrie(int n) : n(n), root(new Node(0, 0)) {}
    void Insert(std::list<size_t>, double, double);
    virtual double GetProb(std::list<size_t>);
};
