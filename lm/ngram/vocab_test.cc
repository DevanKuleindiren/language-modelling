#include "gtest/gtest.h"
#include "vocab.h"


class VocabTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new Vocab();
    }

    Vocab *under_test;
};

TEST_F(VocabTest, CorrectIndexOrder) {
    under_test->Insert("zero");
    under_test->Insert("one");
    under_test->Insert("two");

    EXPECT_EQ(under_test->Get("zero").value, 0);
    EXPECT_EQ(under_test->Get("one").value, 1);
    EXPECT_EQ(under_test->Get("two").value, 2);
    EXPECT_EQ(under_test->Get("blah").has_value, false);

    EXPECT_EQ(under_test->OOVIndex(), 3);
}

TEST_F(VocabTest, HandlesDuplicates) {
    EXPECT_EQ(under_test->Insert("zero"), 0);
    EXPECT_EQ(under_test->Insert("one"), 1);
    EXPECT_EQ(under_test->Insert("zero"), 0);
    EXPECT_EQ(under_test->Insert("two"), 2);
    EXPECT_EQ(under_test->Insert("three"), 3);
    EXPECT_EQ(under_test->Insert("one"), 1);
    EXPECT_EQ(under_test->Insert("three"), 3);
    EXPECT_EQ(under_test->Insert("four"), 4);
    EXPECT_EQ(under_test->Insert("three"), 3);

    EXPECT_EQ(under_test->Get("zero").value, 0);
    EXPECT_EQ(under_test->Get("one").value, 1);
    EXPECT_EQ(under_test->Get("two").value, 2);
    EXPECT_EQ(under_test->Get("three").value, 3);
    EXPECT_EQ(under_test->Get("four").value, 4);
    EXPECT_EQ(under_test->Get("blah").has_value, false);

    EXPECT_EQ(under_test->OOVIndex(), 5);
}

TEST_F(VocabTest, Iterator) {
    under_test->Insert("zero");
    under_test->Insert("one");
    under_test->Insert("two");

    std::unordered_map<std::string, size_t> actual_output;
    std::unordered_map<std::string, size_t> expected_output;
    expected_output.insert(std::make_pair("zero", 0));
    expected_output.insert(std::make_pair("one", 1));
    expected_output.insert(std::make_pair("two", 2));

    std::unordered_map<std::string, size_t>::const_iterator it = under_test->begin();
    for (; it != under_test->end(); ++it) {
        actual_output.insert(*it);
    }

    EXPECT_EQ(actual_output, expected_output);
}
