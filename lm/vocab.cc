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

bool Vocab::ContainsWord(std::string word) {
    return word_to_id.count(word) > 0;
}

void Vocab::ProcessFile(std::string file_name) {
    std::ifstream f (file_name);
    std::unordered_map<std::string, int> word_counts;

    if (f.is_open()) {
        std::string line;

        Insert("<s>");
        while (std::getline(f, line)) {
            std::string word;
            std::istringstream iss (line);
            while (iss >> word) {
                word_counts[word]++;

                if (word_counts[word] == min_frequency) {
                    Insert(word);
                }
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

bool Vocab::operator==(const Vocab &to_compare) {
    return (min_frequency == to_compare.min_frequency) && (id == to_compare.id) && (word_to_id == to_compare.word_to_id);
}

void Vocab::Save(std::string file_name) {
    tensorflow::Source::lm::VocabProto vocab_proto;
    vocab_proto.set_min_frequency(min_frequency);

    for (std::unordered_map<std::string, size_t>::iterator it = word_to_id.begin(); it != word_to_id.end(); ++it) {
        tensorflow::Source::lm::VocabProto::Item *item_proto = vocab_proto.add_item();
        item_proto->set_id(it->second);
        item_proto->set_word(it->first);
    }

    std::ofstream ofs (file_name, std::ios::out | std::ios::trunc);
    google::protobuf::io::OstreamOutputStream osos(&ofs);
    if (!google::protobuf::TextFormat::Print(vocab_proto, &osos)) {
        std::cerr << "Failed to write vocab proto." << std::endl;
    } else {
        std::cout << "Saved vocab proto." << std::endl;
    }
}

Vocab *Vocab::Load(std::string file_name) {
    std::ifstream ifs (file_name, std::ios::in);
    tensorflow::Source::lm::VocabProto vocab_proto;

    google::protobuf::io::IstreamInputStream isis(&ifs);
    if (!google::protobuf::TextFormat::Parse(&isis, &vocab_proto)) {
        std::cerr << "Failed to read vocab trie." << std::endl;
    } else {
        std::cout << "Read vocab proto." << std::endl;
    }

    Vocab *vocab = new Vocab(vocab_proto.min_frequency());
    for (int i = 0; i < vocab_proto.item_size(); ++i) {
        vocab->word_to_id[vocab_proto.item(i).word()] = vocab_proto.item(i).id();
    }
    vocab->id = vocab->word_to_id.size();

    ifs.close();
    return vocab;
}

int Vocab::Size() {
    int size = word_to_id.size();
    if (word_to_id.count("<s>") > 0) size--;
    return size;
}
