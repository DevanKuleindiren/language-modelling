#ifndef file_reader_h
#define file_reader_h

#include <fstream>
#include <list>
#include <sstream>
#include <string>

class FileReader {
    std::ifstream input_file;
    std::istringstream input_line;
public:
    FileReader(std::string file_name) : input_file(std::ifstream(file_name)) {}
    bool GetNextWord(std::string *);
};

#endif // file_reader.h
