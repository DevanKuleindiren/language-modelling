#ifndef prob_trie_h
#define prob_trie_h

#include <fcntl.h>
#include <iostream>
#include <list>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include "tensorflow/Source/lm/ngram/prob_trie.pb.h"

class ProbTrie {
public:
    struct Node {
        double pseudo_prob;
        double backoff;
        std::unordered_map<size_t, Node*> children;
        Node(double pseudo_prob, double backoff) : pseudo_prob(pseudo_prob), backoff(backoff), children() {}
        bool operator==(const Node &);
    };
protected:
    Node *root;
    Node *GetNode(std::list<size_t>);
    static void PopulateProto(tensorflow::Source::lm::ngram::Node *, const ProbTrie::Node *);
    static void PopulateProbTrie(ProbTrie::Node *, const tensorflow::Source::lm::ngram::Node *);
public:
    ProbTrie() : root(new Node(0, 0)) {}
    void Insert(std::list<size_t>, double, double);
    virtual double GetProb(std::list<size_t>);
    virtual std::pair<double, double> GetValues(std::list<size_t>);
    virtual bool operator==(const ProbTrie &);
    virtual tensorflow::Source::lm::ngram::ProbTrieProto *ToProto();
    static ProbTrie *FromProto(const tensorflow::Source::lm::ngram::ProbTrieProto *);
    Node *GetRoot();
};

#endif // prob_trie.h
