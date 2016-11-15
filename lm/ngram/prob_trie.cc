#include "prob_trie.h"


void ProbTrie::Insert(std::list<std::string> seq, double pseudo_prob, double backoff) {
    ProbTrie::Node *curr = root;

    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        if (curr->children.empty() || curr->children.count(*it) < 1) {
            curr->children.insert(std::make_pair(*it, new ProbTrie::Node(0, 1)));
        }
        curr = (curr->children.find(*it))->second;
    }
    curr->pseudo_prob = pseudo_prob;
    curr->backoff = backoff;
}

double ProbTrie::GetProb(std::list<std::string> seq) {
    int seq_size = seq.size();
    if (seq_size > 0) {

        // Ensure sequence contains at most N elements.
        if (seq_size > n) {
            std::list<std::string> tmp;
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
            std::string last_word = seq.back();
            seq.pop_back();
            ProbTrie::Node *backoff_node = GetNode(seq);
            seq.push_back(last_word);

            if (backoff_node != NULL) {
                backoff = backoff_node->backoff;
            }
        }

        seq.pop_front();
        return pseudo_prob + (backoff * GetProb(seq));
    }
    return 0;
}

ProbTrie::Node *ProbTrie::GetNode(std::list<std::string> seq) {
    if (seq.size() > 0) {
        ProbTrie::Node *curr = root;

        for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
            if (curr->children.empty() || curr->children.count(*it) < 1) {
                return NULL;
            }
            curr = (curr->children.find(*it))->second;
        }
        return curr;
    }
    return NULL;
}

double ProbTrie::GetProbRecurse(std::list<std::string> seq) {
    if (seq.size() > 0) {
        ProbTrie::Node *node = GetNode(seq);
        if (node == NULL) {
            return 0;
        } else {
            seq.pop_front();
            return node->backoff * (node->pseudo_prob + GetProbRecurse(seq));
        }
    }
    return 0;
}
