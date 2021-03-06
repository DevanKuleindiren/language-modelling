#include <fstream>
#include <google/protobuf/util/message_differencer.h>
#include "ngram.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/Source/lm/ngram/ngram.pb.h"
#include "tensorflow/Source/lm/ngram/prob_trie.pb.h"
#include "tensorflow/Source/lm/vocab.pb.h"


std::string SetUp() {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the mouse .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();

    return test_file_name;
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
    NGram *under_test = new NGram(::SetUp(), 2, 1);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the"})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat"})), 1/2.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"cat", "ate"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigram) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 0.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"mouse", ".", "<s>"})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({".", "<s>", "the"})), 1.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTestProb, ProbTrigramWithMinFreq) {
    NGram *under_test = new NGram(::SetUp(), 3, 2);

    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "cat"})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"<s>", "the", "frog"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "cat", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"on", "the", "mat"})), 0.5);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "sat"})), 1/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"the", "mouse", "."})), 2/3.0);
    ASSERT_DOUBLE_EQ(under_test->Prob(std::list<std::string>({"blah", "blah", "blah"})), 0.0);
}

TEST(NGramTest, ProbAllFollowing) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);
    std::list<std::pair<std::string, double>> probs;

    under_test->ProbAllFollowing(std::list<std::string>({"<s>", "the"}), probs);
    ASSERT_EQ(probs.size(), 11);
}

TEST(NGramTest, ProbAllFollowingCharTrie) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);

    CharTrie *char_trie = new CharTrie();
    char_trie->Insert("cat", 0.0);
    char_trie->Insert("sat", 0.0);
    char_trie->Insert("dog", 0.0);

    under_test->ProbAllFollowing(std::list<std::string>({"<s>", "the"}), char_trie);

    ASSERT_DOUBLE_EQ(char_trie->GetProb("cat"), 2/3.0);
    ASSERT_DOUBLE_EQ(char_trie->GetProb("dog"), 1/3.0);
    ASSERT_DOUBLE_EQ(char_trie->GetProb("sat"), 0.0);
}

TEST(NGramTest, PredictTrigram) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);
    std::pair<std::string, double> prediction;

    under_test->Predict(std::list<std::string>({"<s>", "the"}), prediction);
    EXPECT_EQ(prediction.first, "cat");
    ASSERT_DOUBLE_EQ(prediction.second, 2/3.0);

    under_test->Predict(std::list<std::string>({"the", "mouse"}), prediction);
    EXPECT_EQ(prediction.first, ".");
    ASSERT_DOUBLE_EQ(prediction.second, 1.0);
}

TEST(NGramTest, PredictTopKTrigram) {
    NGram *under_test = new NGram(::SetUp(), 3, 1);
    std::list<std::pair<std::string, double>> predictions;

    std::list<std::pair<std::string, double>> expected_predictions;
    expected_predictions.push_back(std::make_pair("cat", 2/3.0));
    expected_predictions.push_back(std::make_pair("dog", 1/3.0));

    under_test->PredictTopK(std::list<std::string>({"<s>", "the"}), predictions, 2);

    EXPECT_EQ(predictions, expected_predictions);
}

TEST(NGramTest, EqualsOpTrue) {
    std::string test_file_name = ::SetUp();
    NGram *under_test_a = new NGram(test_file_name, 3, 1);
    NGram *under_test_b = new NGram(test_file_name, 3, 1);

    ASSERT_TRUE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseN) {
    std::string test_file_name = ::SetUp();
    NGram *under_test_a = new NGram(test_file_name, 3, 1);
    NGram *under_test_b = new NGram(test_file_name, 2, 1);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseMinFreq) {
    std::string test_file_name = ::SetUp();
    NGram *under_test_a = new NGram(test_file_name, 3, 1);
    NGram *under_test_b = new NGram(test_file_name, 3, 2);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, EqualsOpFalseDifferentWords) {
    NGram *under_test_a = new NGram(::SetUp(), 3, 1);

    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the cat sat on the mat .\n";
    test_file << "the cat ate the dog .\n";
    test_file << "the dog sat on the cat .\n";
    test_file.close();
    NGram *under_test_b = new NGram(test_file_name, 2, 1);

    ASSERT_FALSE(*under_test_a == *under_test_b);
}

TEST(NGramTest, ToProto) {
    std::ofstream test_file;
    std::string test_file_name = "/tmp/ngram_test_file";
    test_file.open (test_file_name, std::ofstream::out | std::ofstream::trunc);
    test_file << "the the cat\n";
    test_file.close();

    NGram *under_test = new NGram(test_file_name, 2, 1);

    tensorflow::Source::lm::ngram::NGramProto *expected_ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    tensorflow::Source::lm::ngram::NGramProto *actual_ngram_proto = under_test->ToProto();

    /* The expected proto structure:

    a (0, 0)
    --<s>-- b (0, 1)
            --the-- d (1, 0)
    --the-- c (0, 1)
            --the-- e (0.5, 0)
            --cat-- f (0.5, 0)
    */

    expected_ngram_proto->set_n(2);
    expected_ngram_proto->set_smoothing(tensorflow::Source::lm::ngram::Smoothing::NONE);

    tensorflow::Source::lm::ngram::ProbTrieProto *prob_trie_proto = new tensorflow::Source::lm::ngram::ProbTrieProto();

    tensorflow::Source::lm::ngram::Node *node_d = ::BuildNode(1, 0);
    tensorflow::Source::lm::ngram::Node *node_e = ::BuildNode(0.5, 0);
    tensorflow::Source::lm::ngram::Node *node_f = ::BuildNode(0.5, 0);

    tensorflow::Source::lm::ngram::Node *node_b = ::BuildNode(0, 1);
    SetChildProperties(node_b->add_child(), 2, node_d);

    tensorflow::Source::lm::ngram::Node *node_c = ::BuildNode(0, 1);
    SetChildProperties(node_c->add_child(), 2, node_e);
    SetChildProperties(node_c->add_child(), 3, node_f);

    tensorflow::Source::lm::ngram::Node *node_a = ::BuildNode(0, 0);
    SetChildProperties(node_a->add_child(), 1, node_b);
    SetChildProperties(node_a->add_child(), 2, node_c);

    prob_trie_proto->set_allocated_root(node_a);
    expected_ngram_proto->set_allocated_prob_trie(prob_trie_proto);

    ASSERT_TRUE(google::protobuf::util::MessageDifferencer::Equals(*actual_ngram_proto, *expected_ngram_proto));
}

TEST(NGramTest, Save) {
    NGram *under_test = new NGram(::SetUp(), 2, 1);
    under_test->Save("/tmp");

    Vocab *vocab_loaded = Vocab::Load("/tmp/vocab.pbtxt");

    std::ifstream ifs ("/tmp/ngram.pb", std::ios::in | std::ios::binary);
    tensorflow::Source::lm::ngram::NGramProto *ngram_proto = new tensorflow::Source::lm::ngram::NGramProto();
    if (!ngram_proto->ParseFromIstream(&ifs)) {
        std::cerr << "Failed to read ngram proto." << std::endl;
    } else {
        std::cout << "Read ngram proto." << std::endl;
    }
    ifs.close();

    int n_loaded = ngram_proto->n();
    ProbTrie *prob_trie_loaded = ProbTrie::FromProto(&(ngram_proto->prob_trie()));
    NGram *under_test_loaded = new NGram(n_loaded, prob_trie_loaded, vocab_loaded);

    ASSERT_TRUE(*under_test == *under_test_loaded);
}
