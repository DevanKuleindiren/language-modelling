#include "tensorflow/Source/lm/ngram/ngram.h"

class AddOneProbTrie : public ProbTrie {
public:
    AddOneProbTrie(int n) : ProbTrie(n) {}
    virtual double GetProb(std::list<size_t> seq) {
        if (seq.size() >= n) {

            // Ensure sequence contains at most N elements.
            if (seq.size() > n) {
                std::list<size_t> tmp;
                for (int i = 0; i < n; i++) {
                    tmp.push_front(seq.back());
                    seq.pop_back();
                }
                seq = tmp;
            }

            ProbTrie::Node *node = GetNode(seq);
            double count = 0;
            double sum_following = 0;

            if (node != NULL) {
                count = node->pseudo_prob;
            }

            seq.pop_back();
            ProbTrie::Node *sum_following_node = GetNode(seq);
            if (sum_following_node != NULL) {
                sum_following = sum_following_node->backoff;
            }

            return (count + 1) / (sum_following + root->pseudo_prob);
        }
        return 0;
    }
};

class AddOne : public NGram {
protected:
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<size_t>);
public:
    AddOne(int n) : AddOne(n, 1) {}
    AddOne(int n, int min_frequency) : NGram(n, new AddOneProbTrie(n), min_frequency) {}
};
