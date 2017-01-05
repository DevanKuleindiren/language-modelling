#ifndef lm_h
#define lm_h

#include <exception>
#include <list>
#include <string>
#include <utility>
#include "vocab.h"

class LM {
protected:
    Vocab *vocab;
    virtual std::list<size_t> WordsToIndices(std::list<std::string> seq);
    virtual std::list<size_t> Trim(std::list<size_t>, int);
public:
    LM() : vocab(NULL) {}
    LM(int min_frequency) : vocab(new Vocab(min_frequency)) {}
    LM(Vocab *vocab) : vocab(vocab) {}
    virtual bool ContainsWord(std::string);
    virtual std::pair<int, int> ContextSize() = 0;
    virtual void Predict(std::list<std::string>, std::pair<std::string, double> &) = 0;
    virtual void PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int) = 0;
    virtual double Prob (std::list<std::string>) = 0;
};

struct UntrainedException : public std::exception {
    const char* what() const noexcept {
        return "The language model must first be trained or loaded from file before making any predictions.\n";
    }
};

#endif // lm.h
