#include "vocab.h"

size_t Vocab::Insert(std::string word) {
    if (word_to_index.count(word) == 0) {
        word_to_index.insert(std::make_pair(word, index));
        index++;
    }
    return word_to_index.find(word)->second;
}

size_t Vocab::Get(std::string word) {
    if (word_to_index.count(word) == 0) {
        return 0;
    }
    return word_to_index.find(word)->second;
}

void Vocab::ProcessFile(std::string file_name) {
    std::ifstream f (file_name);
    std::unordered_map<std::string, int> word_counts;

    if (f.is_open()) {
        std::string line;

        Insert("<s>");
        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }
                word = line.substr(0, pos);
                word_counts[word]++;

                if (word_counts[word] == min_frequency) {
                    Insert(word);
                }

                line.erase(0, pos + 1);
            }
        }
    }
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::begin() {
    return word_to_index.begin();
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::end() {
    return word_to_index.end();
}
