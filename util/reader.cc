#include "reader.h"

bool FileReader::GetNextWord(std::string *next_word) {
    if (input_line >> *next_word) {
        return true;
    } else {
        std::string line;
        input_line.clear();
        if (std::getline(input_file, line)) {
            input_line << line << std::flush;
            *next_word = "<s>";
            return true;
        } else {
            return false;
        }
    }
}
