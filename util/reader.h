#ifndef file_reader_h
#define file_reader_h

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

class FileReader {
    std::ifstream &input_file;
    std::stringstream input_line;
public:
    FileReader(std::ifstream &input_file) : input_file(input_file), input_line() {}
    bool GetNextWord(std::string *);
};

#endif // file_reader.h
