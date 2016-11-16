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
}
