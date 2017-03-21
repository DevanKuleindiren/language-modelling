#ifndef ngram_h
#define ngram_h

#include <fstream>
#include <list>
#include <map>
#include <string>
#include <vector>
#include <unordered_set>
#include "count_trie.h"
#include "prob_trie.h"
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/ngram/ngram.pb.h"
#include "tensorflow/Source/lm/vocab.h"


class NGram : public LM {
protected:
    const int n;
    ProbTrie *prob_trie;
    NGram(int n, int min_frequency) : LM(min_frequency), n(n), prob_trie(new ProbTrie()) {}
    virtual void ProcessFile(std::string);
    virtual void ProcessCountTrie(CountTrie *);
    virtual void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    NGram(std::string file_name, int n, int min_frequency);
    NGram(int n, ProbTrie *prob_trie, Vocab *vocab) : LM(vocab), n(n), prob_trie(prob_trie) {}
    virtual std::pair<int, int> ContextSize();
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<size_t>);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual void ProbAllFollowing (std::list<std::string>, CharTrie *);
    virtual bool operator==(const NGram &);
    virtual tensorflow::Source::lm::ngram::NGramProto *ToProto();
    virtual void Save(std::string);
};

#endif // ngram.h
