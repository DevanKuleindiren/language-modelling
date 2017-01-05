#include <iostream>
#include <fstream>
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/Source/benchmark/benchmark.h"
#include "tensorflow/Source/benchmark/benchmark.pb.h"
#include "tensorflow/Source/lm/ngram/load.h"
#include "tensorflow/Source/lm/rnn/lstm.h"

#define RNN std::string("rnn")
#define NGRAM std::string("ngram")

void usage(char* const argv_0) {
    std::cerr << "Usage: " << argv_0;
    std::cerr << " --model_path=PATH --type=TYPE --test_data_path=TEST" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    PATH is the path the directory containing the model protos." << std::endl;
    std::cerr << "    TYPE is the type of language model (one of: " << RNN << " or " << NGRAM << ")." << std::endl;
    std::cerr << "    TEST is the file path of the test data to run the benchmarking against." << std::endl;
}

int main(int argc, char* argv[]) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string model_path;
    std::string type;
    std::string test_data_path;

    const bool parse_result = tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("model_path", &model_path),
        tensorflow::Flag("type", &type),
        tensorflow::Flag("test_data_path", &test_data_path),
    });
    if (!parse_result) {
        usage(argv[0]);
        return -1;
    }

    if (model_path.empty()) {
        std::cerr << "Error: --model_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (type.empty()) {
        std::cerr << "Error: --type must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (test_data_path.empty()) {
        std::cerr << "Error: --test_data_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }

    LM *lm;
    if (type.compare(RNN) == 0) {
        lm = new LSTM(model_path);
    } else {
        lm = Load(model_path);
    }

    Benchmark *benchmark = new Benchmark(lm);
    double perplexity = benchmark->Perplexity(test_data_path, false);
    std::cout << "Perplexity = " << perplexity << std::endl;

    tensorflow::Source::benchmark::BenchmarkProto benchmark_proto;
    benchmark_proto.set_perplexity(perplexity);

    if (model_path.back() != '/') {
        model_path += '/';
    }
    std::ofstream ofs (model_path + "benchmark.pbtxt", std::ios::out | std::ios::trunc);
    google::protobuf::io::OstreamOutputStream osos(&ofs);
    if (!google::protobuf::TextFormat::Print(benchmark_proto, &osos)) {
        std::cerr << "Failed to write benchmark proto." << std::endl;
    } else {
        std::cout << "Saved benchmark proto." << std::endl;
    }
}
