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

bool ProbTrie::operator==(const ProbTrie &to_compare) {
    return (n == to_compare.n && *root == *to_compare.root);
}

void ProbTrie::Save(std::string file_name) {
    tensorflow::Source::lm::ngram::ProbTrieProto prob_trie_proto;
    prob_trie_proto.set_n(n);
    PopulateProto(prob_trie_proto.mutable_root(), root);

    std::ofstream ofs (file_name, std::ios::out | std::ios::trunc);
    google::protobuf::io::OstreamOutputStream osos(&ofs);
    if (!google::protobuf::TextFormat::Print(prob_trie_proto, &osos)) {
        std::cerr << "Failed to write prob trie." << std::endl;
    } else {
        std::cout << "Saved prob_trie_proto." << std::endl;
    }
    // TODO: Deallocate heap-allocated proto instance.
}

void ProbTrie::Load(std::string file_name) {
    std::ifstream ifs (file_name, std::ios::in);
    tensorflow::Source::lm::ngram::ProbTrieProto prob_trie_proto;

    google::protobuf::io::IstreamInputStream isis(&ifs);
    if (!google::protobuf::TextFormat::Parse(&isis, &prob_trie_proto)) {
        std::cerr << "Failed to read prob trie." << std::endl;
    } else {
        std::cout << "Read prob_trie_proto." << std::endl;
    }

    n = prob_trie_proto.n();
    PopulateProbTrie(root, &prob_trie_proto.root());
    ifs.close();
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
