#ifndef kneser_ney_h
#define kneser_ney_h

#include "tensorflow/Source/lm/ngram/ngram.h"
#include <algorithm>

class KneserNey : public NGram {
protected:
    double discount;
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    KneserNey(int n, double discount) : KneserNey(n, discount, 1) {}
    KneserNey(int n, double discount, int min_frequency) : NGram(n, min_frequency), discount(discount) {}
    KneserNey(int n, double discount, ProbTrie *prob_trie, Vocab *vocab) : NGram(n, prob_trie, vocab), discount(discount) {}
    virtual bool operator==(const NGram &);
    tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // kneser_ney.h
