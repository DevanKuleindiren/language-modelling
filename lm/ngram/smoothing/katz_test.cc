#include <fstream>
#include <google/protobuf/util/message_differencer.h>
#include "katz.h"
#include "tensorflow/core/platform/test.h"


class KatzTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        std::ofstream test_file;
        std::string test_file_name = "/tmp/katz_test_file";
        test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test = new Katz(test_file_name, 3, 1);
    }
    Katz *under_test;
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

TEST_F(KatzTest, Prob) {
    /* Counts:
    n\r| 0 | 1 | 2 | 3 | 4 | 5 | 6
    -------------------------------
     1 |-1 | 4 | 2 | 3 | 0 | 0 | 1
     2 | 66| 10| 3 | 2 | 0 | 0 | 0
     2 |711| 15| 3 | 0 | 0 | 0 | 0

    r * d_{n,r}:
     n\r| 0   | 1 | 2 | 3 | 4 | 5
     ----------------------------
      1 | 8   | 1 |-3 | 9 | 0 | 0
      2 |5/33 |3/5| 2 | 0 | 0 | 0
      2 |5/237|2/5| 0 | 0 | 0 | 0
    */
    
    // P(cat|<s> the) = 0/3
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 0.0);

    // P(sat|the cat) = (2/5)/3
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 2/15.0);

    // P(mat|on the) = (2/5)/2
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.2);

    // P(sat|the mouse) = ((1-2/5)/(1-3/5))((1-3/5)/(1-9/23))(-3/23)
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), -9/70.0);

    // P(mat|the mouse) = ((1-2/5)/(1-3/5))((1-3/5)/(1-9/23))(1/23)
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "mat"})), 3/70.0);

    // P(mat|the ate) = ((1-0)/(1-0))((1-3/5)/(1-6/23))(1/23)
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "ate", "mat"})), 2/85.0);

    // P(.|the mouse) = (2/5)/1
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 0.4);

    // P(mat|blah blah) = ((1-0)/(1-0))((1-0)/(1-0))(1/23)
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "mat"})), 1/23.0);

    // P(blah|blah blah) = 0
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(KatzTestToProto, ToProto) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/katz_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the the cat\n";
    test_file.close();

    Katz *under_test = new Katz(test_file_name, 2, 1);

    tensorflow::Source::lm::ngram::NGramProto *expected_ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    tensorflow::Source::lm::ngram::NGramProto *actual_ngram_proto = under_test->ToProto();

    /* The expected proto structure:

    a (0, 0)
    --<s>-- b (1/4, 1)
            --the-- e (0, 1)
    --the-- c (0, 4/3)
            --the-- f (0, 1)
            --cat-- g (0, 1)
    --cat-- d (1/4, 1)
    */

    expected_ngram_proto->set_n(2);
    expected_ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KATZ);

    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();

    tensorflow::Source::lm::ngram::Node *node_e = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_f = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_g = ::BuildNode(0, 1);

    tensorflow::Source::lm::ngram::Node *node_b = ::BuildNode(0.25, 1);
    SetChildProperties(node_b->add_child(), 2, node_e);

    tensorflow::Source::lm::ngram::Node *node_c = ::BuildNode(0, 4/3.0);
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
