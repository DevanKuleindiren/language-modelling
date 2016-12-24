#ifndef lstm_h
#define lstm_h

#include <algorithm>
#include <list>
#include <map>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/ngram/vocab.h"


class LSTM : public LM {
protected:
    Vocab *vocab;
    bool trained = false;
    tensorflow::Session *session;
    tensorflow::Status status;
    unsigned long batch_size;
    unsigned long num_steps;
    std::list<size_t> WordsToIndices(std::list<std::string>);
    virtual void RunInference(std::list<size_t>, std::vector<tensorflow::Tensor> &);
public:
    LSTM(std::string, int min_frequency);
    virtual bool ContainsWord(std::string);
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &);
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int);
    virtual double Prob(std::list<std::string>);
};

#endif // lstm.h
