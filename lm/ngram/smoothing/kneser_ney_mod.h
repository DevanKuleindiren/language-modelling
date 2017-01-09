#ifndef kneser_ney_mod_h
#define kneser_ney_mod_h

#include "kneser_ney.h"
#include "tensorflow/Source/lm/ngram/ngram.h"

class KneserNeyMod : public KneserNey {
protected:
    std::vector<std::vector<double>> discounts;
    virtual void ProcessCountTrie(CountTrie *);
    virtual double Discount(int, int);
    virtual double BackoffNumerator(CountTrie *, int, std::list<size_t>);
public:
    KneserNeyMod(std::string file_name, int n, int min_frequency);
    KneserNeyMod(int n, std::list<double> discounts, ProbTrie *prob_trie, Vocab *vocab);
    virtual bool operator==(const NGram &);
    virtual tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // kneser_ney.h
