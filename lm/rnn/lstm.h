#ifndef lstm_h
#define lstm_h

#include <algorithm>
#include <list>
#include <map>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/vocab.h"


class LSTM : public LM {
protected:
    tensorflow::Session *session;
    tensorflow::Status status;
    unsigned long num_steps;
    virtual void RunInference(std::list<size_t>, std::vector<tensorflow::Tensor> &);
public:
    LSTM(std::string);
    virtual std::pair<int, int> ContextSize();
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &);
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int);
    virtual double Prob(std::list<std::string>);
};

#endif // lstm.h
