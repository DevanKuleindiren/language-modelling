#include "count_trie.h"


void CountTrie::ProcessFile(std::string file_name) {
    std::ifstream f (file_name);

    if (f.is_open()) {
        std::string line;
        int line_number = 0;

        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;
            std::list<std::string> ngram_window;

            for (int i = 0; i < n - 1; i++) {
                ngram_window.push_back("<s>");
            }

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }

                word = line.substr(0, pos);

                ngram_window.push_back(word);
                std::list<std::string> ngram;
                for (std::list<std::string>::reverse_iterator it = ngram_window.rbegin(); it != ngram_window.rend(); ++it) {
                    ngram.push_front(*it);
                    Insert(ngram);
                }

                if (ngram_window.size() >= n) {
                    ngram_window.pop_front();
                }

                line.erase(0, pos + 1);
            }

            line_number++;
            if (line_number % 10000 == 0) {
                std::cout << "Read " << line_number << " lines." << std::endl;
            }
        }
        std::list<std::string> seq;
        ComputeCountsAndSums(root, seq);
    }
}

int CountTrie::Count(std::list<std::string> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count;
    }
    return 0;
}

int CountTrie::CountFollowing(std::list<std::string> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_following;
    }
    return 0;
}

int CountTrie::CountPreceding(std::list<std::string> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_preceding;
    }
    return 0;
}

int CountTrie::CountPrecedingAndFollowing(std::list<std::string> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_preceding_and_following;
    }
    return 0;
}

int CountTrie::SumFollowing(std::list<std::string> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->sum_following;
    }
    return 0;
}

void CountTrie::PopulateVocab(std::unordered_set<std::string> *vocab) {
    for (std::unordered_map<std::string, CountTrie::Node*>::iterator it = root->children.begin(); it != root->children.end(); ++it) {
        if ((it->first).compare("<s>") != 0) {
            vocab->insert(it->first);
        }
    }
}

int CountTrie::VocabSize() {
    if (root->children.count("<s>") > 0) {
        return root->count_following - 1;
    }
    return root->count_following;
}

CountTrie::Node *CountTrie::GetNode(std::list<std::string> seq, bool create_new) {
    CountTrie::Node *curr = root;

    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        if (curr->children.count(*it) < 1) {
            if (create_new) {
                curr->children.insert(std::make_pair(*it, new CountTrie::Node()));
            } else {
                return NULL;
            }
        }
        curr = (curr->children.find(*it))->second;
    }
    return curr;
}

void CountTrie::Insert(std::list<std::string> seq) {
    if (seq.size() <= n) {
        CountTrie::Node *curr = root;

        for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
            if (curr->children.empty() || curr->children.count(*it) < 1) {
                curr->children.insert(std::make_pair(*it, new CountTrie::Node()));
            }
            curr = (curr->children.find(*it))->second;
        }

        curr->count++;

        std::string predecessor = seq.front();
        seq.pop_front();
        if (seq.size() > 0) {
            CountTrie::Node *node = GetNode(seq, true);
            if (node->predecessors.count(predecessor) == 0) {
                node->predecessors.insert(predecessor);
                node->count_preceding++;
            }
        }
    }
}

void CountTrie::ComputeCountsAndSums(CountTrie::Node *node, std::list<std::string> seq) {
    int sum_following = 0;
    int count_following = 0;
    int count_preceding_and_following = 0;

    for (auto child = node->children.begin(); child != node->children.end(); ++child) {
        seq.push_back(child->first);
        ComputeCountsAndSums(child->second, seq);
        seq.pop_back();

        sum_following += child->second->count;
        count_following += 1;
        count_preceding_and_following += child->second->count_preceding;
    }

    node->sum_following = sum_following;
    node->count_following = count_following;
    node->count_preceding_and_following = count_preceding_and_following;
}

CountTrie::Node *CountTrie::GetRoot() {
    return root;
}
