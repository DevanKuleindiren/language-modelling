#include "tensorflow/core/platform/test.h"
#include "dual_reader.h"

TEST(DuplicatedDualFileReaderTest, ReadsCorrectly) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/file_reader_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << " the cat sat on the mat . \n";
    test_file << " the cat ate the mouse . \n";
    test_file << " the dog sat on the cat . \n";
    test_file.close();

    DuplicatedDualFileReader *under_test = new DuplicatedDualFileReader(test_file_name);

    std::list<std::string> expected_words ({"<s>", "the", "cat", "sat", "on", "the", "mat", ".",
                                            "<s>", "the", "cat", "ate", "the", "mouse", ".",
                                            "<s>", "the", "dog", "sat", "on", "the", "cat", "."});

    std::string next_input_word;
    std::string next_target_word;

    for (std::list<std::string>::iterator it = expected_words.begin(); it != expected_words.end(); ++it) {
        under_test->GetNextInputTargetWordPair(&next_input_word, &next_target_word);
        ASSERT_STREQ(next_input_word.c_str(), (*it).c_str());
        ASSERT_STREQ(next_target_word.c_str(), (*it).c_str());
    }
}

TEST(DifferentDualFileReaderTest, ReadsCorrectly) {
    std::ofstream test_file_inputs;
    std::string test_file_name_inputs = "/tmp/file_reader_test_file_inputs";
    test_file_inputs.open (test_file_name_inputs, std::ofstream::out | std::ofstream::trunc);
    test_file_inputs << " the cart sat on the mate . \n";
    test_file_inputs << " thhe cat ate the mouse . \n";
    test_file_inputs << " the dog sat in the cat . \n";
    test_file_inputs.close();

    std::ofstream test_file_targets;
    std::string test_file_name_targets = "/tmp/file_reader_test_file_targets";
    test_file_targets.open (test_file_name_targets, std::ofstream::out | std::ofstream::trunc);
    test_file_targets << " the cat sat on the mat . \n";
    test_file_targets << " the cat ate the mouse . \n";
    test_file_targets << " the dog sat on the cat . \n";
    test_file_targets.close();

    DifferentDualFileReader *under_test = new DifferentDualFileReader(test_file_name_inputs, test_file_name_targets);

    std::list<std::string> expected_input_words ({"<s>", "the", "cart", "sat", "on", "the", "mate", ".",
                                                  "<s>", "thhe", "cat", "ate", "the", "mouse", ".",
                                                  "<s>", "the", "dog", "sat", "in", "the", "cat", "."});

    std::list<std::string> expected_target_words ({"<s>", "the", "cat", "sat", "on", "the", "mat", ".",
                                                  "<s>", "the", "cat", "ate", "the", "mouse", ".",
                                                  "<s>", "the", "dog", "sat", "on", "the", "cat", "."});

    std::string next_input_word;
    std::string next_target_word;

    std::list<std::string>::iterator input_it = expected_input_words.begin();
    std::list<std::string>::iterator target_it = expected_target_words.begin();

    for (; input_it != expected_input_words.end() && target_it != expected_target_words.end(); ++input_it, ++target_it) {
        under_test->GetNextInputTargetWordPair(&next_input_word, &next_target_word);
        ASSERT_STREQ(next_input_word.c_str(), (*input_it).c_str());
        ASSERT_STREQ(next_target_word.c_str(), (*target_it).c_str());
    }
}
