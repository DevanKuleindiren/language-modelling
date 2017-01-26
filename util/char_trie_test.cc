#include "char_trie.h"
#include "tensorflow/core/platform/test.h"


class CharTrieTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new CharTrie();
        under_test->Insert("the", 0.4);
        under_test->Insert("their", 0.35);
        under_test->Insert("they", 0.38);
        under_test->Insert("car", 0.2);
        under_test->Insert("carpet", 0.1);
        under_test->Insert("carton", 0.55);
        under_test->Insert("cop", 0.26);
        under_test->Insert("candle", 0.64);
        under_test->Insert("then", 0.63);
        under_test->Insert("thesaurus", 0.02);
        under_test->Insert("rope", 0.11);
        under_test->Insert("and", 0.78);
        under_test->Insert("them", 0.56);
        under_test->Insert("too", 0.7);
        under_test->Insert("told", 0.001);
        under_test->Insert("theme", 0.21);
    }

    CharTrie *under_test;
};

// Note: In practice, these probabilities would sum to 1.
TEST_F(CharTrieTest, Update) {
    ASSERT_STREQ(under_test->GetMaxWithPrefix("").first.c_str(), "and");
    ASSERT_EQ(under_test->GetMaxWithPrefix("").second, 0.78);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("t").first.c_str(), "too");
    ASSERT_EQ(under_test->GetMaxWithPrefix("t").second, 0.7);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("to").first.c_str(), "too");
    ASSERT_EQ(under_test->GetMaxWithPrefix("to").second, 0.7);

    under_test->Update("candle", 0.8);
    under_test->Update("thesaurus", 0.72);

    ASSERT_STREQ(under_test->GetMaxWithPrefix("").first.c_str(), "candle");
    ASSERT_EQ(under_test->GetMaxWithPrefix("").second, 0.8);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("t").first.c_str(), "thesaurus");
    ASSERT_EQ(under_test->GetMaxWithPrefix("t").second, 0.72);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("to").first.c_str(), "too");
    ASSERT_EQ(under_test->GetMaxWithPrefix("to").second, 0.7);
}

TEST_F(CharTrieTest, GetMaxWithPrefix) {
    ASSERT_STREQ(under_test->GetMaxWithPrefix("").first.c_str(), "and");
    ASSERT_EQ(under_test->GetMaxWithPrefix("").second, 0.78);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("t").first.c_str(), "too");
    ASSERT_EQ(under_test->GetMaxWithPrefix("t").second, 0.7);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("to").first.c_str(), "too");
    ASSERT_EQ(under_test->GetMaxWithPrefix("to").second, 0.7);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("th").first.c_str(), "then");
    ASSERT_EQ(under_test->GetMaxWithPrefix("th").second, 0.63);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("the").first.c_str(), "then");
    ASSERT_EQ(under_test->GetMaxWithPrefix("the").second, 0.63);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("c").first.c_str(), "candle");
    ASSERT_EQ(under_test->GetMaxWithPrefix("c").second, 0.64);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("car").first.c_str(), "carton");
    ASSERT_EQ(under_test->GetMaxWithPrefix("car").second, 0.55);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("told").first.c_str(), "told");
    ASSERT_EQ(under_test->GetMaxWithPrefix("told").second, 0.001);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("toldt").first.c_str(), "");
    ASSERT_EQ(under_test->GetMaxWithPrefix("toldt").second, 0);
    ASSERT_STREQ(under_test->GetMaxWithPrefix("blah").first.c_str(), "");
    ASSERT_EQ(under_test->GetMaxWithPrefix("blah").second, 0);
}

TEST_F(CharTrieTest, GetMaxKWithPrefix) {
    std::list<std::pair<std::string, double>> expected_top = {
        {"and", 0.78},
        {"too", 0.7},
        {"candle", 0.64},
        {"then", 0.63},
        {"them", 0.56},
    };
    std::list<std::pair<std::string, double>> actual_top = under_test->GetMaxKWithPrefix("", 5);
    ASSERT_EQ(actual_top, expected_top);

    expected_top.clear();
    expected_top.push_back(std::make_pair("then", 0.63));
    expected_top.push_back(std::make_pair("them", 0.56));
    expected_top.push_back(std::make_pair("the", 0.4));
    actual_top = under_test->GetMaxKWithPrefix("th", 3);
    ASSERT_EQ(actual_top, expected_top);

    expected_top.clear();
    expected_top.push_back(std::make_pair("carton", 0.55));
    expected_top.push_back(std::make_pair("car", 0.2));
    expected_top.push_back(std::make_pair("carpet", 0.1));
    actual_top = under_test->GetMaxKWithPrefix("car", 3);
    ASSERT_EQ(actual_top, expected_top);

    expected_top.clear();
    expected_top.push_back(std::make_pair("thesaurus", 0.02));
    actual_top = under_test->GetMaxKWithPrefix("thes", 3);
    ASSERT_EQ(actual_top, expected_top);

    expected_top.clear();
    actual_top = under_test->GetMaxKWithPrefix("blah", 3);
    ASSERT_EQ(actual_top, expected_top);
}

TEST_F(CharTrieTest, Contains) {
    ASSERT_TRUE(under_test->Contains("the"));
    ASSERT_TRUE(under_test->Contains("tol"));
    ASSERT_TRUE(under_test->Contains("thesau"));
    ASSERT_FALSE(under_test->Contains("z"));
    ASSERT_FALSE(under_test->Contains("blah"));
}
