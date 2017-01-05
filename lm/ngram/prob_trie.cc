#include "prob_trie.h"


bool ProbTrie::Node::operator==(const ProbTrie::Node &to_compare) {

    if (pseudo_prob != to_compare.pseudo_prob ||
        backoff != to_compare.backoff ||
        children.size() != to_compare.children.size()) {
        return false;
    }

    for (std::unordered_map<size_t, ProbTrie::Node*>::iterator it = children.begin(); it != children.end(); ++it) {
        if (to_compare.children.count(it->first) == 0 ||
            !((*(to_compare.children.find(it->first))->second) == *(it->second))) {
            return false;
        }
    }

    return true;
}

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

std::pair<double, double> ProbTrie::GetValues(std::list<size_t> seq) {
    ProbTrie::Node *node = GetNode(seq);

    if (node == NULL) {
        return std::make_pair(0, 0);
    } else {
        return std::make_pair(node->pseudo_prob, node->backoff);
    }
}

bool ProbTrie::Contains(std::list<size_t> seq) {
    return GetNode(seq) != NULL;
}

bool ProbTrie::operator==(const ProbTrie &to_compare) {
    return *root == *to_compare.root;
}

tensorflow::Source::lm::ngram::ProbTrieProto *ProbTrie::ToProto() {
    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();
    PopulateProto(prob_trie_proto->mutable_root(), root);
    return prob_trie_proto;
}

ProbTrie *ProbTrie::FromProto(const tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto) {
    ProbTrie *prob_trie = new ProbTrie();
    PopulateProbTrie(prob_trie->root, &(prob_trie_proto->root()));
    return prob_trie;
}

ProbTrie::Node *ProbTrie::GetRoot() {
    return root;
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

void ProbTrie::PopulateProto(tensorflow::Source::lm::ngram::Node *node_proto, const ProbTrie::Node *node) {
    node_proto->set_pseudo_prob(node->pseudo_prob);
    node_proto->set_backoff(node->backoff);
    for (std::unordered_map<size_t, ProbTrie::Node*>::const_iterator it = node->children.begin(); it != node->children.end(); ++it) {
        tensorflow::Source::lm::ngram::Node::Child *child_proto = node_proto->add_child();
        child_proto->set_id(it->first);
        PopulateProto(child_proto->mutable_node(), it->second);
    }
}

void ProbTrie::PopulateProbTrie(ProbTrie::Node *node, const tensorflow::Source::lm::ngram::Node *node_proto) {
    node->pseudo_prob = node_proto->pseudo_prob();
    node->backoff = node_proto->backoff();
    for (int i = 0; i < node_proto->child_size(); ++i) {
        ProbTrie::Node *child_node = new ProbTrie::Node(0, 1);
        node->children.insert(std::make_pair(node_proto->child(i).id(), child_node));
        PopulateProbTrie(child_node, &node_proto->child(i).node());
    }
}
