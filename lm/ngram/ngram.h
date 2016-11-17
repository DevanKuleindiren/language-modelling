#ifndef ngram_h
#define ngram_h

#include "lm/lm.h"
#include "prob_trie.h"
#include "count_trie.h"
#include "vocab.h"
#include <fstream>
#include <list>
#include <map>
#include <string>
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
    NGram(int n) : NGram(n, new ProbTrie(n)) {}
    NGram(int n, ProbTrie *prob_trie) : n(n), prob_trie(prob_trie), vocab(new Vocab()) {}
    virtual bool ContainsWord(std::string);
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &);
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual double Prob(std::list<std::string>);
    virtual void ProcessFile(std::string file_name);
};

#endif // ngram.h
