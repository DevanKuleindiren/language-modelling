#include "count_trie.h"


void CountTrie::ProcessFile(std::string file_name, Vocab *vocab) {
    FileReader *file_reader = new FileReader(file_name);
    int num_words = 0;
    std::list<size_t> ngram_window;
    std::string word;

    while (file_reader->GetNextWord(&word)) {
        size_t word_index = vocab->Get(word);
        ngram_window.push_back(word_index);
        InsertSubNGrams(ngram_window);
        ngram_window = Trim(ngram_window, n - 1);

        num_words++;
        if (num_words % 100000 == 0) {
            std::cout << "Read " << num_words << " words." << std::endl;
        }
    }

    std::list<size_t> seq;
    ComputeCountsAndSums(root, seq);
}

int CountTrie::Count(std::list<size_t> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count;
    }
    return 0;
}

int CountTrie::CountFollowing(std::list<size_t> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_following;
    }
    return 0;
}

int CountTrie::CountFollowing(std::list<size_t> seq, int count, bool is_min) {
    CountTrie::Node *node = GetNode(seq, false);
    int result = 0;
    if (node) {
        for (std::unordered_map<size_t, Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            if (is_min && (it->second)->count >= count) {
                result++;
            } else if (!is_min && (it->second)->count == count) {
                result++;
            }
        }
    }
    return result;
}

int CountTrie::CountPreceding(std::list<size_t> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_preceding;
    }
    return 0;
}

int CountTrie::CountPrecedingAndFollowing(std::list<size_t> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->count_preceding_and_following;
    }
    return 0;
}

int CountTrie::SumFollowing(std::list<size_t> seq) {
    CountTrie::Node *node = GetNode(seq, false);
    if (node) {
        return node->sum_following;
    }
    return 0;
}

int CountTrie::VocabSize() {
    // Minus 1 because of the <s> marker.
    return root->count_following - 1;
}

CountTrie::Node *CountTrie::GetNode(std::list<size_t> seq, bool create_new) {
    CountTrie::Node *curr = root;

    for (std::list<size_t>::iterator it = seq.begin(); it != seq.end(); ++it) {
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

void CountTrie::Insert(std::list<size_t> seq) {
    if (seq.size() <= n) {
        CountTrie::Node *curr = root;

        for (std::list<size_t>::iterator it = seq.begin(); it != seq.end(); ++it) {
            if (curr->children.empty() || curr->children.count(*it) < 1) {
                curr->children.insert(std::make_pair(*it, new CountTrie::Node()));
            }
            curr = (curr->children.find(*it))->second;
        }

        curr->count++;

        size_t predecessor = seq.front();
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

void CountTrie::ComputeCountsAndSums(CountTrie::Node *node, std::list<size_t> seq) {
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

void CountTrie::CountNGrams(std::vector<std::vector<int>> *n_r) {
    CountNGramsRec(root, 0, n_r);
}

void CountTrie::CountNGramsRec(CountTrie::Node *node, int depth, std::vector<std::vector<int>> *n_r) {
    int max_n = (*n_r).size() - 1;
    int max_k = (*n_r)[0].size() - 1;

    if (depth > max_n) return;

    if (depth > 0) {
        (*n_r)[depth][0]++;
        if (node->count <= max_k) {
            (*n_r)[depth][node->count]++;
        }
    }

    for (std::unordered_map<size_t, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
        CountNGramsRec(it->second, depth + 1, n_r);
    }

    if (depth == 0) {
        for (int i = 1; i <= max_n; ++i) {
            (*n_r)[i][0] = (int) (pow(VocabSize(), i) - (*n_r)[i][0]);
        }
    }
}

void CountTrie::InsertSubNGrams(std::list<size_t> ngram_window) {
    std::list<size_t> ngram;
    for (std::list<size_t>::reverse_iterator it = ngram_window.rbegin(); it != ngram_window.rend(); ++it) {
        ngram.push_front(*it);
        Insert(ngram);
    }
}

std::list<size_t> CountTrie::Trim(std::list<size_t> seq, int max) {
    while (seq.size() > max) {
        seq.pop_front();
    }
    return seq;
}
