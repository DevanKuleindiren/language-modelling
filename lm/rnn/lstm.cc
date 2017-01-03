#include "lstm.h"

LSTM::LSTM(std::string directory_path) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    vocab = Vocab::Load(directory_path + "/vocab.pbtxt");

    status = NewSession(tensorflow::SessionOptions(), &session);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    tensorflow::GraphDef graph_def;
    status = ReadBinaryProto(tensorflow::Env::Default(), directory_path + "graph.pb", &graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }

    for (int i = 0; i < graph_def.node_size(); i++) {
        if (graph_def.node(i).name().compare("inference/lstm/inputs") == 0) {
            num_steps = graph_def.node(i).attr().at("shape").shape().dim(1).size();
        }
    }

    status = session->Create(graph_def);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}

bool LSTM::ContainsWord(std::string word) {
    return vocab->Get(word);
}

std::pair<int, int> LSTM::ContextSize() {
    return std::make_pair(1, num_steps);
}

void LSTM::Predict(std::list<std::string> seq, std::pair<std::string, double> &prediction) {
    std::list<size_t> seq_ids = WordsToIndices(seq);
    seq_ids = Trim(seq_ids, ContextSize().second);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs);
    auto predictions = outputs[0].tensor<float, 2>();

    double max_prob = 0;
    std::string max_prediction;
    int last_word_position = seq_ids.size() - 1;
    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        if (predictions(last_word_position, it->second) > max_prob) {
            max_prob = predictions(last_word_position, it->second);
            max_prediction = it->first;
        }
    }

    prediction = std::make_pair(max_prediction, max_prob);
}

void LSTM::PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int) {
    // TODO: Implement this.
}

double LSTM::Prob(std::list<std::string> seq) {
    std::list<size_t> seq_ids = WordsToIndices(seq);
    seq_ids = Trim(seq_ids, ContextSize().second);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs);
    auto predictions = outputs[0].tensor<float, 2>();

    return predictions(seq_ids.size() - 2, seq_ids.back());
}

std::list<size_t> LSTM::WordsToIndices(std::list<std::string> seq) {
    std::list<size_t> indices;
    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        indices.push_back(vocab->Get(*it));
    }
    return indices;
}

std::list<size_t> LSTM::Trim(std::list<size_t> seq, int max) {
    if (seq.size() > max) {
        std::list<size_t> trimmed;
        for (int i = 0; i < max; i++) {
            trimmed.push_front(seq.back());
            seq.pop_back();
        }
        return trimmed;
    }
    return seq;
}

void LSTM::RunInference(std::list<size_t> seq_ids, std::vector<tensorflow::Tensor> &outputs) {
    // Ensure there are at most num_steps words in the sequence.
    if (seq_ids.size() > num_steps) {
        int diff = seq_ids.size() - num_steps;
        for (int i = 0; i < diff; ++i) {
            seq_ids.pop_front();
        }
    }

    // Create and populate the LSTM input tensor.
    tensorflow::Tensor seq_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({1, static_cast<long long>(num_steps)}));
    auto seq_tensor_raw = seq_tensor.tensor<int, 2>();
    for (int step = 0; step < num_steps; ++step) {
        seq_tensor_raw(0, step) = 0;
    }
    int i = 0;
    for (std::list<size_t>::iterator it = seq_ids.begin(); it != seq_ids.end(); ++it) {
        seq_tensor_raw(0, i) = (int) *it;
        i++;
    }

    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
        {"inference/lstm/inputs", seq_tensor},
    };

    // Run the inference to the node named 'inference/lstm/predictions'.
    session->Run(inputs, {"inference/lstm/predictions"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}
