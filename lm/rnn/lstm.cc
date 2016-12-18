#include "lstm.h"

LSTM::LSTM(std::string directory_path, int min_frequency) {
    if (directory_path.back() != '/') {
        directory_path += '/';
    }

    vocab = new Vocab(min_frequency);
    vocab->Load(directory_path + "/vocab.pbtxt");

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

    trained = true;
}

bool LSTM::ContainsWord(std::string word) {
    return vocab->Get(word);
}

void LSTM::Predict(std::list<std::string> seq, std::pair<std::string, double> &prediction) {
    if (!trained) {
        throw UntrainedException();
    }
    std::list<size_t> seq_ids = WordsToIndices(seq);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs);
    auto predictions = outputs[0].tensor<float, 2>();

    double max_prob = 0;
    std::string max_prediction;
    int last_word_position = std::max(seq.size(), NUM_STEPS) - 1;
    for (std::unordered_map<std::string, size_t>::const_iterator it = vocab->begin(); it != vocab->end(); ++it) {
        if (predictions(last_word_position, it->second) > max_prob) {
            max_prob = predictions(last_word_position, it->second);
            max_prediction = it->first;
        }
    }

    prediction = std::make_pair(max_prediction, max_prob);
}

void LSTM::PredictTopK(std::list<std::string>, std::list<std::pair<std::string, double>> &, int) {
    if (!trained) {
        throw UntrainedException();
    }
    // TODO: Implement this.
}

double LSTM::Prob(std::list<std::string> seq) {
    if (!trained) {
        throw UntrainedException();
    }
    std::list<size_t> seq_ids = WordsToIndices(seq);

    std::vector<tensorflow::Tensor> outputs;
    RunInference(seq_ids, outputs);
    auto predictions = outputs[0].tensor<float, 2>();

    return predictions(std::max(seq.size(), NUM_STEPS) - 2, seq_ids.back());
}

std::list<size_t> LSTM::WordsToIndices(std::list<std::string> seq) {
    std::list<size_t> indices;
    for (std::list<std::string>::iterator it = seq.begin(); it != seq.end(); ++it) {
        indices.push_back(vocab->Get(*it));
    }
    return indices;
}

void LSTM::RunInference(std::list<size_t> seq_ids, std::vector<tensorflow::Tensor> &outputs) {
    // Ensure there are at most NUM_STEPS words in the sequence.
    if (seq_ids.size() > NUM_STEPS) {
        int diff = seq_ids.size() - NUM_STEPS;
        for (int i = 0; i < diff; ++i) {
            seq_ids.pop_front();
        }
    }

    // Create and populate the LSTM input tensor.
    tensorflow::Tensor seq_tensor(tensorflow::DT_INT32, tensorflow::TensorShape({BATCH_SIZE, NUM_STEPS}));
    auto seq_tensor_raw = seq_tensor.tensor<int, 2>();
    for (int i = 0; i < BATCH_SIZE; ++i) {
        for (int j = 0; j < BATCH_SIZE; ++j) {
            seq_tensor_raw(i, j) = 0;
        }
    }
    int i = 0;
    for (std::list<size_t>::iterator it = seq_ids.begin(); it != seq_ids.end(); ++it) {
        seq_tensor_raw(0, i) = (int) *it;
        i++;
    }

    std::vector<std::pair<tensorflow::string, tensorflow::Tensor>> inputs = {
        {"Model/inputs", seq_tensor},
    };

    // Run the inference to the node named 'Model/predictions'.
    session->Run(inputs, {"Model/predictions"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
    }
}
