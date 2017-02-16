#ifndef dual_reader_h
#define dual_reader_h

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>
#include "reader.h"

class DualFileReader {
public:
    virtual bool GetNextInputTargetWordPair(std::string *, std::string *) = 0;
};

class DuplicatedDualFileReader : public DualFileReader {
public:
    FileReader *file_reader;
public:
    DuplicatedDualFileReader(std::string input_file_name) : file_reader(new FileReader(input_file_name)) {}
    bool GetNextInputTargetWordPair(std::string *input, std::string *target) {
        std::string next_word;
        if (file_reader->GetNextWord(&next_word)) {
            *input = next_word;
            *target = next_word;
            return true;
        } else {
            return false;
        }
    };
};

class DifferentDualFileReader : public DualFileReader {
public:
    FileReader *input_file_reader;
    FileReader *target_file_reader;
public:
    DifferentDualFileReader(std::string input_file_name, std::string target_file_name) :
        input_file_reader(new FileReader(input_file_name)), target_file_reader(new FileReader(target_file_name)) {}
    bool GetNextInputTargetWordPair(std::string *input, std::string *target) {
        std::string next_input_word;
        std::string next_target_word;
        if (input_file_reader->GetNextWord(&next_input_word) && target_file_reader->GetNextWord(&next_target_word)) {
            *input = next_input_word;
            *target = next_target_word;
            return true;
        } else {
            return false;
        }
    };
};

#endif // dual_reader.h
