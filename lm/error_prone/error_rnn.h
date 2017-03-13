#ifndef error_rnn_h
#define error_rnn_h

#include <algorithm>
#include <assert.h>
#include <list>
#include <map>
#include <string>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/Source/lm/lm.h"
#include "tensorflow/Source/lm/vocab.h"
#include "tensorflow/Source/lm/rnn/rnn.h"
#include "tensorflow/Source/util/char_trie.h"
#include "tensorflow/Source/util/prediction_compare.h"


class ErrorCorrectingRNN : public RNN {
protected:
    std::set<std::string> dict;
    tensorflow::Tensor previous_predictions;
    virtual void RunCorrection(std::string, std::vector<tensorflow::Tensor> &, std::string, bool);
    virtual void RunCorrections(std::list<std::string>, std::vector<tensorflow::Tensor> &, std::string, bool);
public:
    ErrorCorrectingRNN(std::string, std::string);
    virtual double Prob(std::list<std::string>);
    virtual double Prob(std::list<std::string>, bool);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual void ProbAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &, bool);
    virtual void ProbAllFollowing (std::list<std::string>, CharTrie *);
    virtual void ProbAllFollowing (std::list<std::string>, CharTrie *, bool);
    virtual void LogitsAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &);
    virtual void LogitsAllFollowing (std::list<std::string>, std::list<std::pair<std::string, double>> &, bool);
    virtual void LogitsAllFollowing (std::list<std::string>, CharTrie *);
    virtual void LogitsAllFollowing (std::list<std::string>, CharTrie *, bool);
    virtual int EditDistance(std::string, std::string);
};

#endif // error_rnn.h
