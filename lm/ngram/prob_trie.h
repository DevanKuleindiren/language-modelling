#ifndef prob_trie_h
#define prob_trie_h

#include <fcntl.h>
#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <list>
#include <stdio.h>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include "tensorflow/Source/lm/ngram/prob_trie.pb.h"

#endif /* prob_trie_h */


class ProbTrie {
protected:
    struct Node {
        double pseudo_prob;
        double backoff;
        std::unordered_map<size_t, Node*> children;
        Node(double pseudo_prob, double backoff) : pseudo_prob(pseudo_prob), backoff(backoff), children() {}
        bool operator==(const Node &);
    };
    Node *root;
    Node *GetNode(std::list<size_t>);
    void PopulateProto(tensorflow::Source::lm::ngram::Node *, const ProbTrie::Node *);
    void PopulateProbTrie(ProbTrie::Node *, const tensorflow::Source::lm::ngram::Node *);
public:
    ProbTrie() : root(new Node(0, 0)) {}
    void Insert(std::list<size_t>, double, double);
    virtual double GetProb(std::list<size_t>);
    virtual bool operator==(const ProbTrie &);
    virtual void Save(std::string file_name);
    virtual void Load(std::string file_name);
};
