#include "lm/ngram/ngram.h"

class AddOneProbTrie : public ProbTrie {
public:
    AddOneProbTrie(int n) : ProbTrie(n) {}
    virtual double GetProb(std::list<std::string> seq) {
        if (seq.size() >= n) {

            // Ensure sequence contains at most N elements.
            if (seq.size() > n) {
                std::list<std::string> tmp;
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
    void PopulateProbTrie(CountTrie *, CountTrie::Node *, int, std::list<std::string>);
public:
    AddOne(int n) : NGram(n, new AddOneProbTrie(n)) {}
};
