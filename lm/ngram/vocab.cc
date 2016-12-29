#include "vocab.h"

size_t Vocab::Insert(std::string word) {
    if (word_to_id.count(word) == 0) {
        word_to_id.insert(std::make_pair(word, id));
        id++;
    }
    return word_to_id.find(word)->second;
}

size_t Vocab::Get(std::string word) {
    if (word_to_id.count(word) == 0) {
        return 0;
    }
    return word_to_id.find(word)->second;
}

void Vocab::ProcessFile(std::string file_name) {
    std::ifstream f (file_name);
    std::unordered_map<std::string, int> word_counts;

    if (f.is_open()) {
        std::string line;

        Insert("<s>");
        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }
                word = line.substr(0, pos);
                word_counts[word]++;

                if (word_counts[word] == min_frequency) {
                    Insert(word);
                }

                line.erase(0, pos + 1);
            }
        }
    }
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::begin() {
    return word_to_id.begin();
}

std::unordered_map<std::string, size_t>::const_iterator Vocab::end() {
    return word_to_id.end();
}

tensorflow::Source::lm::VocabProto *Vocab::ToProto() {
    tensorflow::Source::lm::VocabProto *vocab_proto = new tensorflow::Source::lm::VocabProto();
    vocab_proto->set_min_frequency(min_frequency);

    for (std::unordered_map<std::string, size_t>::iterator it = word_to_id.begin(); it != word_to_id.end(); ++it) {
        tensorflow::Source::lm::VocabProto::Item *item_proto = vocab_proto->add_item();
        item_proto->set_id(it->second);
        item_proto->set_word(it->first);
    }

    return vocab_proto;
}

Vocab *Vocab::FromProto(tensorflow::Source::lm::VocabProto *vocab_proto) {
    Vocab *vocab = new Vocab(vocab_proto->min_frequency());
    for (int i = 0; i < vocab_proto->item_size(); ++i) {
        vocab->word_to_id[vocab_proto->item(i).word()] = vocab_proto->item(i).id();
    }
    return vocab;
}

int Vocab::Size() {
    return word_to_id.size();
}
