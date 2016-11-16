#include <experimental/optional>
#include <string>
#include <unordered_map>
#include <utility>

template <class T> struct Optional {
    T value;
    bool has_value;
    Optional() : value(), has_value(false) {}
    Optional(T value) : value(value), has_value(true) {}
};

class Vocab {
    size_t index;
    std::unordered_map<std::string, size_t> word_to_index;
public:
    Vocab() : index(0) {}
    size_t Insert(std::string word);
    Optional<size_t> Get(std::string word);
};
