#ifndef count_trie_h
#define count_trie_h

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include "tensorflow/Source/lm/vocab.h"
#include "tensorflow/Source/util/reader.h"

class CountTrie {
public:
    struct Node {
        int count;
        int sum_following;
        int count_following;
        int count_preceding;
        int count_preceding_and_following;
        std::unordered_map<size_t, Node*> children;
        std::unordered_set<size_t> predecessors;
    };
private:
    Node *root;
    int n;
    Node *GetNode(std::list<size_t>, bool);
    void CountNGramsRec(Node *, int, std::vector<std::vector<int>> *);
    void InsertSubNGrams(std::list<size_t>);
    std::list<size_t> Trim(std::list<size_t>, int);
public:
    CountTrie(int n) : root(new Node()), n(n) {}
    void ProcessFile(std::string, Vocab *);
    int Count(std::list<size_t>);
    int CountFollowing(std::list<size_t>);
    int CountFollowing(std::list<size_t>, int, bool);
    int CountPreceding(std::list<size_t>);
    int CountPrecedingAndFollowing(std::list<size_t>);
    int SumFollowing(std::list<size_t>);
    int VocabSize();
    void Insert(std::list<size_t>);
    void ComputeCountsAndSums(Node *, std::list<size_t>);
    Node *GetRoot();
    void CountNGrams(std::vector<std::vector<int>> *);
};

#endif /* count_trie_h */
