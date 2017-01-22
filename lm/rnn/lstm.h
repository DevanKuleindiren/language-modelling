#ifndef lstm_h
#define lstm_h

#include <algorithm>
#include <assert.h>
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
    std::vector<tensorflow::Tensor> state;
    virtual void ResetState();
    virtual void RunInference(size_t, std::vector<tensorflow::Tensor> &, bool);
    virtual void RunInference(std::list<size_t>, std::vector<tensorflow::Tensor> &, bool);
public:
    LSTM(std::string);
    virtual std::pair<int, int> ContextSize();
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<std::string>, bool);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &, bool);
};

#endif // lstm.h
