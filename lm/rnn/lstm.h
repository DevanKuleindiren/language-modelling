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

// TODO: Encode this data into the LSTM protobufs rather than hard-coding it.
#define BATCH_SIZE 10ul
#define NUM_STEPS 10ul

class LSTM : public LM {
protected:
    Vocab *vocab;
    bool trained = false;
    tensorflow::Session *session;
    tensorflow::Status status;
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
