#ifndef ngram_h
#define ngram_h

#include "lm/lm.h"
#include "prob_trie.h"
#include "count_trie.h"
#include <fstream>
#include <list>
#include <map>
#include <string>
#include <unordered_set>


class NGram : public LM {
protected:
    int n;
    ProbTrie *prob_trie;
    std::unordered_set<std::string> vocab;
    bool trained = false;
    virtual void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<std::string>);
public:
    NGram(int n) : n(n), prob_trie(new ProbTrie(n)), vocab() {}
    NGram(int n, ProbTrie *prob_trie) : n(n), prob_trie(prob_trie), vocab() {}
    virtual bool ContainsWord(std::string);
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &);
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual double Prob(std::list<std::string>);
    virtual void ProcessFile(std::string file_name);
};

#endif // ngram.h
