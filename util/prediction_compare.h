#ifndef prediction_compare_h
#define prediction_compare_h

#include <string>
#include <utility>

class PredictionCompare {
public:
    bool operator() (std::pair<std::string, double> const &a, std::pair<std::string, double> const &b) const {
        if (a.second == b.second) {
            return a.first > b.first;
        }
        return a.second > b.second;
    }
};

#endif // prediction_compare.h
