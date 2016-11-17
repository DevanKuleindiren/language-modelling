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

size_t Vocab::OOVIndex() {
    return index;
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::begin() {
    return word_to_index.begin();
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::end() {
    return word_to_index.end();
}
