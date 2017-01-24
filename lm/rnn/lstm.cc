#include "lstm.h"

#define INPUT_NAME "inference/lstm/inputs"
#define PREDICTION_NAME "inference/lstm/predictions"

#define INITIAL_C_0_NAME "inference/lstm/zeros"
#define INITIAL_H_0_NAME "inference/lstm/zeros_1"
#define INITIAL_C_1_NAME "inference/lstm/zeros_2"
#define INITIAL_H_1_NAME "inference/lstm/zeros_3"

#define FINAL_C_0_NAME "inference/lstm/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2"
#define FINAL_H_0_NAME "inference/lstm/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2"
#define FINAL_C_1_NAME "inference/lstm/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2"
#define FINAL_H_1_NAME "inference/lstm/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2"


LSTM::LSTM(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    vocab = Vocab::Load(directory_path + "vocab.pbtxt");

    status = NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), directory_path + "graph.pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    ResetState();
}

std::pair<int, int> LSTM::ContextSize() {
    return std::make_pair(1, 2);
}

double LSTM::Prob(std::list<std::string> seq) {
    return Prob(seq, true);
}

double LSTM::Prob(std::list<std::string> seq, bool use_prev_state) {
    std::list<size_t> seq_ids = WordsToIds(seq);
    size_t next_word_id = seq_ids.back();
    seq_ids.pop_back();

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    return predictions(0, next_word_id);
}

void LSTM::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
    ProbAllFollowing(seq, probs, true);
}

void LSTM::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs, bool use_prev_state) {
    std::list<size_t> seq_ids = WordsToIds(seq);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        probs.push_back(std::make_pair(it->first, predictions(0, it->second)));
    }
}

void LSTM::ResetState() {
    // Initialise the input state.
    session->Run({}, {INITIAL_C_0_NAME, INITIAL_H_0_NAME, INITIAL_C_1_NAME, INITIAL_H_1_NAME}, {}, &state);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}

void LSTM::RunInference(size_t seq_id, std::vector<tensorflow::Tensor> &outputs, bool use_prev_state) {
    // Create and populate the LSTM input tensor.
    tensorflow::Tensor seq_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, 1}));
    auto seq_tensor_raw = seq_tensor.tensor<int, 2>();
    seq_tensor_raw(0, 0) = (int) seq_id;

    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {{INPUT_NAME, seq_tensor}};
    if (use_prev_state) {
        inputs.push_back(std::make_pair(INITIAL_C_0_NAME, state[0]));
        inputs.push_back(std::make_pair(INITIAL_H_0_NAME, state[1]));
        inputs.push_back(std::make_pair(INITIAL_C_1_NAME, state[2]));
        inputs.push_back(std::make_pair(INITIAL_H_1_NAME, state[3]));
    } else {
        ResetState();
    }

    // Run the inference to the node named 'inference/lstm/predictions'.
    session->Run(inputs, {PREDICTION_NAME, FINAL_C_0_NAME, FINAL_H_0_NAME, FINAL_C_1_NAME, FINAL_H_1_NAME}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    state[0] = outputs[1];
    state[1] = outputs[2];
    state[2] = outputs[3];
    state[3] = outputs[4];
}

void LSTM::RunInference(std::list<size_t> seq_ids, std::vector<tensorflow::Tensor> &outputs, bool use_prev_state) {
    assert(seq_ids.size() > 0);
    RunInference(seq_ids.front(), outputs, use_prev_state);
    seq_ids.pop_front();
    for (std::list<size_t>::iterator it = seq_ids.begin(); it != seq_ids.end(); ++it) {
        RunInference(*it, outputs, true);
    }
}
