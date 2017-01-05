#include "lm.h"

bool LM::ContainsWord(std::string word) {
    return vocab->Get(word);
}

std::list<size_t> LM::WordsToIndices(std::list<std::string> seq) {
    std::list<size_t> indices;
    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        indices.push_back(vocab->Get(*it));
    }
    return indices;
}

std::list<size_t> LM::Trim(std::list<size_t> seq, int max) {
    if (seq.size() > max) {
        std::list<size_t> trimmed;
        for (int i = 0; i < max; i++) {
            trimmed.push_front(seq.back());
            seq.pop_back();
        }
        return trimmed;
    }
    return seq;
}
