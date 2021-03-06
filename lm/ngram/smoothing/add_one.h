#ifndef add_one_h
#define add_one_h

#include "tensorflow/Source/lm/ngram/ngram.h"


class AddOne : public NGram {
protected:
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    AddOne(std::string file_name, int n, int min_frequency);
    AddOne(int n, ProbTrie *prob_trie, Vocab *vocab) : NGram(n, prob_trie, vocab) {}
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<size_t>);
    tensorflow::Source::lm::ngram::NGramProto *ToProto();
};

#endif // add_one.h
