#include "prob_trie.h"


void ProbTrie::Insert(std::list<size_t> seq, double pseudo_prob, double backoff) {
    ProbTrie::Node *curr = root;

    for (std::list<size_t>::iterator it = seq.begin(); it != seq.end(); ++it) {
        if (curr->children.empty() || curr->children.count(*it) < 1) {
            curr->children.insert(std::make_pair(*it, new ProbTrie::Node(0, 1)));
        }
        curr = (curr->children.find(*it))->second;
    }
    curr->pseudo_prob = pseudo_prob;
    curr->backoff = backoff;
}

double ProbTrie::GetProb(std::list<size_t> seq) {
    int seq_size = seq.size();
    if (seq_size > 0) {

        // Ensure sequence contains at most N elements.
        if (seq_size > n) {
            std::list<size_t> tmp;
            for (int i = 0; i < n; i++) {
                tmp.push_front(seq.back());
                seq.pop_back();
            }
            seq = tmp;
        }

        ProbTrie::Node *node = GetNode(seq);
        double pseudo_prob = 0;
        double backoff = 1;

        if (node != NULL) {
            pseudo_prob = node->pseudo_prob;
        }

        if (seq_size > 1) {
            size_t last_word_index = seq.back();
            seq.pop_back();
            ProbTrie::Node *backoff_node = GetNode(seq);
            seq.push_back(last_word_index);

            if (backoff_node != NULL) {
                backoff = backoff_node->backoff;
            }
        }

        seq.pop_front();
        return pseudo_prob + (backoff * GetProb(seq));
    }
    return 0;
}

ProbTrie::Node *ProbTrie::GetNode(std::list<size_t> seq) {
    if (seq.size() > 0) {
        ProbTrie::Node *curr = root;

        for (std::list<size_t>::iterator it = seq.begin(); it != seq.end(); ++it) {
            if (curr->children.empty() || curr->children.count(*it) < 1) {
                return NULL;
            }
            curr = (curr->children.find(*it))->second;
        }
        return curr;
    }
    return NULL;
}
