#ifndef ngram_h
#define ngram_h

#include "tensorflow/Source/lm/lm.h"
#include "prob_trie.h"
#include "count_trie.h"
#include "vocab.h"
#include <fstream>
#include <list>
#include <map>
#include <queue>
#include <string>
#include <vector>
#include <unordered_set>


class NGram : public LM {
protected:
    int n;
    ProbTrie *prob_trie;
    Vocab *vocab;
    bool trained = false;
    virtual void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
    virtual std::list<size_t> WordsToIndices(std::list<std::string> seq);
public:
    NGram(int n) : NGram(n, new ProbTrie(n), 1) {}
    NGram(int n, int min_frequency) : NGram(n, new ProbTrie(n), min_frequency) {}
    NGram(int n, ProbTrie *prob_trie) : NGram(n, prob_trie, 1) {}
    NGram(int n, ProbTrie *prob_trie, int min_frequency) : n(n), prob_trie(prob_trie), vocab(new Vocab(min_frequency)) {}
    virtual bool ContainsWord(std::string);
    virtual int ContextSize();
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &);
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int);
    virtual double Prob(std::list<std::string>);
    virtual void ProcessFile(std::string file_name);
    virtual void Save(std::string file_name);
    virtual void Load(std::string file_name);
};

class PredictionCompare {
public:
    bool operator() (std::pair<std::string, double> const &a, std::pair<std::string, double> const &b) const {
        if (a.second == b.second) {
            return a.first > b.first;
        }
        return a.second > b.second;
    }
};

#endif // ngram.h
