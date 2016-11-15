#include "add_one.h"

void AddOne::PopulateProbTrie(CountTrie *countTrie, CountTrie::Node *node, int depth, std::list<std::string> seq) {
    if (depth == 0) {
        prob_trie->Insert(seq, countTrie->VocabSize(), 0);
    }

    if (depth < n) {
        if (depth == n - 1) {
            prob_trie->Insert(seq, 0, countTrie->SumFollowing(seq));
        }

        for (std::unordered_map<std::string, CountTrie::Node*>::iterator it = node->children.begin(); it != node->children.end(); ++it) {
            seq.push_back(it->first);
            PopulateProbTrie(countTrie, it->second, depth + 1, seq);
            seq.pop_back();
        }
    } else if (depth == n) {
        prob_trie->Insert(seq, countTrie->Count(seq), 0);
    }
}
