#include "tensorflow/core/platform/test.h"
#include "reader.h"

TEST(FileReaderTest, ReadsCorrectly) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/file_reader_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << " the cat sat on the mat . \n";
    test_file << " the cat ate the mouse . \n";
    test_file << " the dog sat on the cat . \n";
    test_file.close();

    FileReader *under_test = new FileReader(test_file_name);

    std::list<std::string> expected_words ({"<s>", "the", "cat", "sat", "on", "the", "mat", ".",
                                            "<s>", "the", "cat", "ate", "the", "mouse", ".",
                                            "<s>", "the", "dog", "sat", "on", "the", "cat", "."});

    std::string next_word;

    for (std::list<std::string>::iterator it = expected_words.begin(); it != expected_words.end(); ++it) {
        under_test->GetNextWord(&next_word);
        ASSERT_STREQ(next_word.c_str(), (*it).c_str());
    }
}
