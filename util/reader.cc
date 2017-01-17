#include "reader.h"

bool FileReader::GetNextWord(std::string *next_word) {
    if (input_line >> *next_word) {
        return true;
    } else {
        std::string line;
        if (std::getline(input_file, line)) {
            input_line = std::istringstream(line);
            *next_word = "<s>";
            return true;
        } else {
            return false;
        }
    }
}
