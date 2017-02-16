#ifndef reader_h
#define reader_h

#include <fstream>
#include <iostream>
#include <list>
#include <sstream>
#include <string>

class FileReader {
    std::ifstream *input_file;
    std::stringstream input_line;
public:
    FileReader(std::string input_file_name) : input_file(new std::ifstream(input_file_name)), input_line() {}
    bool GetNextWord(std::string *);
};

#endif // reader.h
