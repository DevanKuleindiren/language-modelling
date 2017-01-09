#ifndef kneser_ney_h
#define kneser_ney_h

#include "tensorflow/Source/lm/ngram/ngram.h"
#include <algorithm>

class KneserNey : public NGram {
protected:
    double discount;
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    KneserNey(std::string file_name, int n, double discount, int min_frequency);
    KneserNey(int n, double discount, ProbTrie *prob_trie, Vocab *vocab) : NGram(n, prob_trie, vocab), discount(discount) {}
    virtual std::pair<int, int> ContextSize();
    virtual bool operator==(const NGram &);
    tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // kneser_ney.h
