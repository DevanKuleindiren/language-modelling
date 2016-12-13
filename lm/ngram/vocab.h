#ifndef vocab_h
#define vocab_h

#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

class Vocab {
    int min_frequency;
    size_t index;
    std::unordered_map<std::string, size_t> word_to_index;
    size_t Insert(std::string word);
public:
    Vocab(int min_frequency) : min_frequency(min_frequency), index(1) {
        word_to_index.insert(std::make_pair("<unk>", 0));
    }
    size_t Get(std::string word);
    void ProcessFile(std::string file_name);
    std::unordered_map<std::string, size_t>::const_iterator begin();
    std::unordered_map<std::string, size_t>::const_iterator end();
};

#endif // vocab.h
