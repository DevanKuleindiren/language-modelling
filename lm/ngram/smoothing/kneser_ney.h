#include "lm/ngram/ngram.h"
#include <algorithm>

class KneserNey : public NGram {
protected:
    double discount;
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    KneserNey(int n, double discount) : NGram(n), discount(discount) {}
};
