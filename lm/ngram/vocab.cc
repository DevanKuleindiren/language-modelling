#include "vocab.h"

size_t Vocab::Insert(std::string word) {
    if (word_to_index.count(word) == 0) {
        word_to_index.insert(std::make_pair(word, index));
        index++;
    }
    return word_to_index.find(word)->second;
}

Optional<size_t> Vocab::Get(std::string word) {
    if (word_to_index.count(word) == 0) {
        return Optional<size_t>();
    }
    return Optional<size_t>(word_to_index.find(word)->second);
}
