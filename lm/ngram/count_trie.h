#ifndef count_trie_h
#define count_trie_h

#include <fstream>
#include <iostream>
#include <list>
#include <stdio.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#endif /* count_trie_h */


class CountTrie {
public:
    struct Node {
        int count;
        int sum_following;
        int count_following;
        int count_preceding;
        int count_preceding_and_following;
        std::unordered_map<std::string, Node*> children;
        std::unordered_set<std::string> predecessors;
    };
private:
    Node *root;
    int n;
    Node *GetNode(std::list<std::string>, bool);
public:
    CountTrie(int n) : root(new Node()), n(n) {}
    void ProcessFile(std::string file_name);
    int Count(std::list<std::string>);
    int CountFollowing(std::list<std::string>);
    int CountPreceding(std::list<std::string>);
    int CountPrecedingAndFollowing(std::list<std::string>);
    int SumFollowing(std::list<std::string>);
    void PopulateVocab(std::unordered_set<std::string>*);
    int VocabSize();
    void Insert(std::list<std::string>);
    void ComputeCountsAndSums(Node *node, std::list<std::string>);
    Node *GetRoot();
};
