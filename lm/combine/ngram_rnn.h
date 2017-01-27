#ifndef ngram_rnn_h
#define ngram_rnn_h

#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/ngram/ngram.h"
#include "tensorflow/Source/lm/rnn/rnn.h"


class NGramRNN : public LM {
protected:
    NGram *ngram_lm;
    RNN *rnn_lm;
    std::list<std::string> prev_words;
    virtual double CombineFunction(double, double) = 0;
public:
    NGramRNN(NGram *ngram_lm, RNN *rnn_lm) : LM(ngram_lm->GetVocab()), ngram_lm(ngram_lm), rnn_lm(rnn_lm) {}
    virtual bool ContainsWord(std::string);
    virtual std::pair<int, int> ContextSize();
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<std::string>, bool);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &, bool);
};

class NGramRNNAverage : public NGramRNN {
protected:
    double CombineFunction (double a, double b) {
        return (a + b) / 2.0;
    }
public:
    NGramRNNAverage(NGram *ngram_lm, RNN *rnn_lm) : NGramRNN(ngram_lm, rnn_lm) {}
};

class NGramRNNMax : public NGramRNN {
protected:
    double CombineFunction (double a, double b) {
        return std::max(a, b);
    }
public:
    NGramRNNMax(NGram *ngram_lm, RNN *rnn_lm) : NGramRNN(ngram_lm, rnn_lm) {}
};

class NGramRNNInterp : public NGramRNN {
protected:
    double lambda;
    double CombineFunction (double a, double b) {
        return (lambda * a) + ((1 - lambda) * b);
    }
public:
    NGramRNNInterp(NGram *ngram_lm, RNN *rnn_lm, double lambda) : NGramRNN(ngram_lm, rnn_lm), lambda(lambda) {}
};

#endif // ngram_rnn.h
