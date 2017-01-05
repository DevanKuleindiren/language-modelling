#ifndef katz_h
#define katz_h

#include <algorithm>
#include <array>
#include <math.h>
#include "tensorflow/Source/lm/ngram/ngram.h"

class Katz : public NGram {
protected:
    static const int k = 5;
    void CountNGrams(CountTrie::Node *, int, std::vector<std::vector<int>> *);
    void PopulateProbTriePseudoProb(CountTrie *, CountTrie::Node *, std::vector<std::vector<double>> *, int, std::list<size_t>);
    void PopulateProbTrieBackoff(ProbTrie *, ProbTrie::Node *, int, std::list<size_t>);
public:
    Katz(int n) : Katz(n, 1) {}
    Katz(int n, int min_frequency) : NGram(n, min_frequency) {}
    Katz(int n, ProbTrie *prob_trie, Vocab *vocab) : NGram(n, prob_trie, vocab) {}
    virtual std::pair<int, int> ContextSize();
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<size_t>);
    virtual void ProcessFile(std::string);
    tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // kneser_ney.h
