#include <fstream>
#include <google/protobuf/util/message_differencer.h>
#include "katz.h"
#include "tensorflow/core/platform/test.h"


class KatzTest : public ::testing::Test {
protected:
    virtual void SetUp() {
        under_test = new Katz(3);

        std::ofstream test_file;
        std::string test_file_name = "/tmp/katz_test_file";
        test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
        test_file << "the cat sat on the mat .\n";
        test_file << "the cat ate the mouse .\n";
        test_file << "the dog sat on the cat .\n";
        test_file.close();

        under_test->ProcessFile(test_file_name);
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
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 0.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 4/39.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 2/13.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 0.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "mat"})), 9/143.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "ate", "mat"})), 3/70.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 4/13.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "mat"})), 1/20.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(KatzTestToProto, ToProto) {
    Katz *under_test = new Katz(2, 1);

    std::ofstream test_file;
    std::string test_file_name = "/tmp/katz_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the the cat\n";
    test_file.close();

    under_test->ProcessFile(test_file_name);

    tensorflow::Source::lm::ngram::NGramProto *expected_ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    tensorflow::Source::lm::ngram::NGramProto *actual_ngram_proto = under_test->ToProto();

    /* The expected proto structure:

    a (0, 0)
    --the-- b (0, 3)
            --the-- e (0, 1)
            --cat-- f (0, 1)
    --<s>-- c (0, 1)
            --the-- g (0, 1)
    --cat-- d (2/3, 1)
    */

    expected_ngram_proto->set_n(2);
    expected_ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::KATZ);

    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();

    tensorflow::Source::lm::ngram::Node *node_e = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_f = ::BuildNode(0, 1);
    tensorflow::Source::lm::ngram::Node *node_g = ::BuildNode(0, 1);

    tensorflow::Source::lm::ngram::Node *node_b = ::BuildNode(0, 3);
    SetChildProperties(node_b->add_child(), 2, node_e);
    SetChildProperties(node_b->add_child(), 3, node_f);

    tensorflow::Source::lm::ngram::Node *node_c = ::BuildNode(0, 1);
    SetChildProperties(node_c->add_child(), 2, node_g);

    tensorflow::Source::lm::ngram::Node *node_d = ::BuildNode(2/3.0, 1);

    tensorflow::Source::lm::ngram::Node *node_a = ::BuildNode(0, 0);
    SetChildProperties(node_a->add_child(), 2, node_b);
    SetChildProperties(node_a->add_child(), 1, node_c);
    SetChildProperties(node_a->add_child(), 3, node_d);

    prob_trie_proto->set_allocated_root(node_a);
    expected_ngram_proto->set_allocated_prob_trie(prob_trie_proto);

    ASSERT_TRUE(google::protobuf::util::MessageDifferencer::ApproximatelyEquals(*actual_ngram_proto, *expected_ngram_proto));
}
