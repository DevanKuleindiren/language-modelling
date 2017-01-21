#include "absolute_discounting.h"
#include "tensorflow/core/platform/test.h"
#include <fstream>
#include <google/protobuf/util/message_differencer.h>


class AbsoluteDiscountingTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        std::ofstream test_file;
        std::string test_file_name = "/tmp/absolute_discounting_test_file";
        test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test = new AbsoluteDiscounting(test_file_name, 3, 1);
    }
    AbsoluteDiscounting *under_test;
};

void SetChildProperties(tensorflow::Source::lm::ngram::Node::Child *child, int id, tensorflow::Source::lm::ngram::Node *node) {
    child->set_id(id);
    child->set_allocated_node(node);
}

tensorflow::Source::lm::ngram::Node *BuildNode(double pseudo_prob, double backoff) {
    tensorflow::Source::lm::ngram::Node *node = new tensorflow::Source::lm::ngram::Node();
    node->set_pseudo_prob(pseudo_prob);
    node->set_backoff(backoff);
    return node;
}

TEST_F(AbsoluteDiscountingTest, Prob) {
    // Discount, d = n_1 / (n_1 + 2n_2) = 15/21 = 5/7.
    // P(cat|<s> the) = (2-d)/3 + ((dx2)/3)((3-d/6) + (dx4/6)(3/23))
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 6487/10143.0);

    // P(sat|the cat) = (1-d)/3 + ((dx3)/3)((1-d/3) + (dx3/3)(2/23))
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 234/1127.0);

    // P(mat|on the) = (1-d)/2 + ((dx2)/2)((1-d/6) + (dx4/6)(1/23))
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 216/1127.0);

    // P(sat|the mouse) = (0)/1 + ((dx1)/1)((0/1) + (dx1/1)(2/23))
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 50/1127.0);

    // P(.|the mouse) = (1-d)/1 + ((dx1)/1)((1-d/1) + (dx1/1)(3/23))
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 627/1127.0);

    // P(blah|blah blah) = 0
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(AbsoluteDiscountingTestToProto, ToProto) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/absolute_discounting_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the the cat\n";
    test_file.close();

    AbsoluteDiscounting *under_test = new AbsoluteDiscounting(test_file_name, 2, 1);

    tensorflow::Source::lm::ngram::NGramProto *expected_ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    tensorflow::Source::lm::ngram::NGramProto *actual_ngram_proto = under_test->ToProto();

    /* The expected proto structure:

    a (0, 0)
    --<s>-- b (0.25, 1)
            --the-- e (0, 1)
    --the-- c (0.5, 1)
            --the-- f (0, 1)
            --cat-- g (0, 1)
    --cat-- d (0.25, 0)
    */

    expected_ngram_proto->set_n(2);
    expected_ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::ABSOLUTE_DISCOUNTING);
    expected_ngram_proto->add_discount(1);

    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();

    tensorflow::Source::lm::ngram::Node *node_e = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_f = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_g = ::BuildNode(0, 1);

    tensorflow::Source::lm::ngram::Node *node_b = ::BuildNode(0.25, 1);
    SetChildProperties(node_b->add_child(), 2, node_e);

    tensorflow::Source::lm::ngram::Node *node_c = ::BuildNode(0.5, 1);
    SetChildProperties(node_c->add_child(), 2, node_f);
    SetChildProperties(node_c->add_child(), 3, node_g);

    tensorflow::Source::lm::ngram::Node *node_d = ::BuildNode(0.25, 1);

    tensorflow::Source::lm::ngram::Node *node_a = ::BuildNode(0, 0);
    SetChildProperties(node_a->add_child(), 1, node_b);
    SetChildProperties(node_a->add_child(), 2, node_c);
    SetChildProperties(node_a->add_child(), 3, node_d);

    prob_trie_proto->set_allocated_root(node_a);
    expected_ngram_proto->set_allocated_prob_trie(prob_trie_proto);

    ASSERT_TRUE(google::protobuf::util::MessageDifferencer::ApproximatelyEquals(*actual_ngram_proto, *expected_ngram_proto));
}
