#include <fstream>
#include <google/protobuf/util/message_differencer.h>
#include "ngram.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/Source/lm/ngram/ngram.pb.h"
#include "tensorflow/Source/lm/ngram/prob_trie.pb.h"
#include "tensorflow/Source/lm/vocab.pb.h"


void SetUp(NGram *under_test) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the mouse .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();

    under_test->ProcessFile(test_file_name);
}

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

void SetVocabItemProperties(tensorflow::Source::lm::VocabProto::Item *item, int id, std::string word) {
    item->set_id(id);
    item->set_word(word);
}

TEST(NGramTestProb, ProbBigram) {
    NGram *under_test = new NGram(2, 1);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the"})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat"})), 1/2.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"cat", "ate"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 0.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigramWithMinFreq) {
    NGram *under_test = new NGram(3, 2);
    ::SetUp(under_test);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTest, PredictTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);
    std::pair<std::string, double> prediction;

    under_test->Predict(std::list<std::string>({"<s>", "the"}), prediction);
    EXPECT_EQ(prediction.first, "cat");
    ASSERT_DOUBLE_EQ(prediction.second, 2/3.0);

    under_test->Predict(std::list<std::string>({"the", "mouse"}), prediction);
    EXPECT_EQ(prediction.first, ".");
    ASSERT_DOUBLE_EQ(prediction.second, 1.0);
}

TEST(NGramTest, PredictTopKTrigram) {
    NGram *under_test = new NGram(3, 1);
    ::SetUp(under_test);
    std::list<std::pair<std::string, double>> predictions;

    std::list<std::pair<std::string, double>> expected_predictions;
    expected_predictions.push_back(std::make_pair("cat", 2/3.0));
    expected_predictions.push_back(std::make_pair("dog", 1/3.0));

    under_test->PredictTopK(std::list<std::string>({"<s>", "the"}), predictions, 2);

    EXPECT_EQ(predictions, expected_predictions);
}

TEST(NGramTest, EqualsOpTrue) {
    NGram *under_test_a = new NGram(3, 1);
    ::SetUp(under_test_a);

    NGram *under_test_b = new NGram(3, 1);
    ::SetUp(under_test_b);

    ASSERT_TRUE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseN) {
    NGram *under_test_a = new NGram(3, 1);
    ::SetUp(under_test_a);

    NGram *under_test_b = new NGram(2, 1);
    ::SetUp(under_test_b);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseMinFreq) {
    NGram *under_test_a = new NGram(3, 1);
    ::SetUp(under_test_a);

    NGram *under_test_b = new NGram(3, 2);
    ::SetUp(under_test_b);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseDifferentWords) {
    NGram *under_test_a = new NGram(3, 1);
    ::SetUp(under_test_a);

    NGram *under_test_b = new NGram(2, 1);
    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the dog .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();
    under_test_b->ProcessFile(test_file_name);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, ToProto) {
    NGram *under_test = new NGram(2, 1);

    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the the cat\n";
    test_file.close();

    under_test->ProcessFile(test_file_name);

    tensorflow::Source::lm::ngram::NGramProto *expected_ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    tensorflow::Source::lm::ngram::NGramProto *actual_ngram_proto = under_test->ToProto();

    /* The expected proto structure:

    a (0, 0)
    --the-- b (0, 1)
            --the-- d (0.5, 0)
            --cat-- e (0.5, 0)
    --<s>-- c (0, 1)
            --the-- f (1, 0)
    */

    expected_ngram_proto->set_n(2);
    expected_ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::NONE);

    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();

    tensorflow::Source::lm::ngram::Node *node_d = ::BuildNode(0.5, 0);
    tensorflow::Source::lm::ngram::Node *node_e = ::BuildNode(0.5, 0);
    tensorflow::Source::lm::ngram::Node *node_f = ::BuildNode(1, 0);

    tensorflow::Source::lm::ngram::Node *node_b = ::BuildNode(0, 1);
    SetChildProperties(node_b->add_child(), 2, node_d);
    SetChildProperties(node_b->add_child(), 3, node_e);

    tensorflow::Source::lm::ngram::Node *node_c = ::BuildNode(0, 1);
    SetChildProperties(node_c->add_child(), 2, node_f);

    tensorflow::Source::lm::ngram::Node *node_a = ::BuildNode(0, 0);
    SetChildProperties(node_a->add_child(), 2, node_b);
    SetChildProperties(node_a->add_child(), 1, node_c);

    prob_trie_proto->set_allocated_root(node_a);
    expected_ngram_proto->set_allocated_prob_trie(prob_trie_proto);

    ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(*actual_ngram_proto, *expected_ngram_proto));
}

TEST(NGramTest, Save) {
    NGram *under_test = new NGram(2, 1);
    ::SetUp(under_test);
    under_test->Save("/tmp");

    Vocab *vocab_loaded = Vocab::Load("/tmp/vocab.pbtxt");

    std::ifstream ifs ("/tmp/ngram.pbtxt", std::ios::in);
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();

    google::protobuf::io::IstreamInputStream isis(&ifs);
    google::protobuf::TextFormat::Parse(&isis, ngram_proto);
    ifs.close();

    int n_loaded = ngram_proto->n();
    ProbTrie *prob_trie_loaded = ProbTrie::FromProto(&(ngram_proto->prob_trie()));
    NGram *under_test_loaded = new NGram(n_loaded, prob_trie_loaded, vocab_loaded);

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
