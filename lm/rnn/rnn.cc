#include "rnn.h"


RNN::RNN(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    std::ifstream ifs (directory_path + "rnn.pbtxt", std::ios::in);
    tensorflow::Source::lm::rnn::RNNProto rnn_proto;

    google::protobuf::io::IstreamInputStream isis(&ifs);
    if (!google::protobuf::TextFormat::Parse(&isis, &rnn_proto)) {
        std::cerr << "Failed to read rnn proto." << std::endl;
    } else {
        std::cout << "Read rnn proto." << std::endl;
    }
    ifs.close();

    switch (rnn_proto.type()) {
        case tensorflow::Source::lm::rnn::Type::VANILLA: {
            type = RNN::Type::VANILLA;
            break;
        }
        case tensorflow::Source::lm::rnn::Type::GRU: {
            type = RNN::Type::GRU;
            break;
        }
        case tensorflow::Source::lm::rnn::Type::LSTM: {
            type = RNN::Type::LSTM;
            break;
        }
        default: {
            type = RNN::Type::LSTM;
            break;
        }
    }

    input_tensor_name = rnn_proto.input_tensor_name();
    predictions_tensor_name = rnn_proto.predictions_tensor_name();
    for (int i = 0; i < rnn_proto.h_size(); ++i) {
        tensorflow::Source::lm::rnn::RNNProto::TensorNamePair h_pair = rnn_proto.h(i);
        state_tensor_names.push_back(std::make_pair(h_pair.initial(), h_pair.final()));
    }
    if (type == RNN::Type::LSTM) {
        for (int i = 0; i < rnn_proto.c_size(); ++i) {
            tensorflow::Source::lm::rnn::RNNProto::TensorNamePair c_pair = rnn_proto.c(i);
            state_tensor_names.push_back(std::make_pair(c_pair.initial(), c_pair.final()));
        }
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

std::pair<int, int> RNN::ContextSize() {
    return std::make_pair(1, 2);
}

double RNN::Prob(std::list<std::string> seq) {
    return Prob(seq, true);
}

double RNN::Prob(std::list<std::string> seq, bool use_prev_state) {
    std::list<size_t> seq_ids = WordsToIds(seq);
    size_t next_word_id = seq_ids.back();
    seq_ids.pop_back();

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    return predictions(0, next_word_id);
}

void RNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs) {
    ProbAllFollowing(seq, probs, true);
}

void RNN::ProbAllFollowing (std::list<std::string> seq, std::list<std::pair<std::string, double>> &probs, bool use_prev_state) {
    std::list<size_t> seq_ids = WordsToIds(seq);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs, use_prev_state);
    auto predictions = outputs[0].tensor<float, 2>();

    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        probs.push_back(std::make_pair(it->first, predictions(0, it->second)));
    }
}

void RNN::ResetState() {
    // Initialise the input state.
    std::vector<tensorflow::string> output_tensor_names;
    for (std::list<std::pair<std::string, std::string>>::iterator it = state_tensor_names.begin(); it != state_tensor_names.end(); ++it) {
        output_tensor_names.push_back(it->first);
    }
    session->Run({}, output_tensor_names, {}, &state);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}

void RNN::RunInference(size_t seq_id, std::vector<tensorflow::Tensor> &outputs, bool use_prev_state) {
    // Create and populate the RNN input tensor.
    tensorflow::Tensor seq_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, 1}));
    auto seq_tensor_raw = seq_tensor.tensor<int, 2>();
    seq_tensor_raw(0, 0) = (int) seq_id;

    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {{input_tensor_name, seq_tensor}};
    if (use_prev_state) {
        int i = 0;
        for (std::list<std::pair<std::string, std::string>>::iterator it = state_tensor_names.begin(); it != state_tensor_names.end(); ++it) {
            inputs.push_back(std::make_pair(it->first, state[i]));
            i++;
        }
    } else {
        ResetState();
    }

    // Run the inference to the softmax predictions node.
    std::vector<tensorflow::string> output_tensor_names;
    output_tensor_names.push_back(predictions_tensor_name);
    for (std::list<std::pair<std::string, std::string>>::iterator it = state_tensor_names.begin(); it != state_tensor_names.end(); ++it) {
        output_tensor_names.push_back(it->second);
    }

    session->Run(inputs, output_tensor_names, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    int i = 0;
    for (std::list<std::pair<std::string, std::string>>::iterator it = state_tensor_names.begin(); it != state_tensor_names.end(); ++it) {
        state[i] = outputs[i+1];
        i++;
    }
}

void RNN::RunInference(std::list<size_t> seq_ids, std::vector<tensorflow::Tensor> &outputs, bool use_prev_state) {
    assert(seq_ids.size() > 0);
    RunInference(seq_ids.front(), outputs, use_prev_state);
    seq_ids.pop_front();
    for (std::list<size_t>::iterator it = seq_ids.begin(); it != seq_ids.end(); ++it) {
        RunInference(*it, outputs, true);
    }
}
