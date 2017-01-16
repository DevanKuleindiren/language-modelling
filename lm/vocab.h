#ifndef vocab_h
#define vocab_h

#include <fstream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <iostream>
#include <string>
#include <unordered_map>
#include <utility>
#include "tensorflow/Source/lm/vocab.pb.h"

class Vocab {
    int min_frequency;
    size_t id;
    std::unordered_map<std::string, size_t> word_to_id;
    size_t Insert(std::string word);
public:
    Vocab(int min_frequency) : min_frequency(min_frequency), id(1) {
        word_to_id.insert(std::make_pair("<unk>", 0));
    }
    size_t Get(std::string word);
    bool ContainsWord(std::string word);
    void ProcessFile(std::string file_name);
    std::unordered_map<std::string, size_t>::const_iterator begin();
    std::unordered_map<std::string, size_t>::const_iterator end();
    virtual bool operator==(const Vocab &);
    void Save(std::string);
    static Vocab *Load(std::string);
    int Size();
};

#endif // vocab.h
