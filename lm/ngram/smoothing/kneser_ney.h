#ifndef kneser_ney_h
#define kneser_ney_h

#include <algorithm>
#include "absolute_discounting.h"
#include "tensorflow/Source/lm/ngram/ngram.h"

class KneserNey : public AbsoluteDiscounting {
protected:
    KneserNey(int n, int min_frequency) : AbsoluteDiscounting(n, min_frequency) {}
    virtual void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
    virtual double Discount(int, int);
    virtual double BackoffNumerator(CountTrie *, int, std::list<size_t>);
public:
    KneserNey(std::string file_name, int n, int min_frequency);
    KneserNey(int n, double discount, ProbTrie *prob_trie, Vocab *vocab) : AbsoluteDiscounting(n, discount, prob_trie, vocab) {}
    virtual tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // kneser_ney.h
