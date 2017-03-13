#include <chrono>
#include <iostream>
#include <fstream>
#include <time.h>
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"
#include "tensorflow/Source/benchmark/benchmark.h"
#include "tensorflow/Source/benchmark/benchmark.pb.h"
#include "tensorflow/Source/lm/combine/ngram_rnn.h"
#include "tensorflow/Source/lm/error_prone/error_rnn.h"
#include "tensorflow/Source/lm/ngram/load.h"
#include "tensorflow/Source/lm/rnn/rnn.h"

#define COMB_AVR std::string("avr")
#define COMB_MAX std::string("max")

void usage(char* const argv_0) {
    std::cerr << "Usage: " << argv_0;
    std::cerr << " --ngram_path=PATH --rnn_path=PATH --test_input_path=INPT --test_target_path=TARG";
    std::cerr << " --save_path=SAVE --comb_type=TYPE --error_rnn=BOOL" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    PATH is the path the directory containing the model protos." << std::endl;
    std::cerr << "    INPT is the file path of the input data to run the benchmarking against." << std::endl;
    std::cerr << "    TARG is the file path of the target data to run the benchmarking against." << std::endl;
    std::cerr << "    SAVE is the path the directory to save the benchmark proto." << std::endl;
    std::cerr << "    TYPE is the type of combination used for the language models (one of: ";
    std::cerr << COMB_AVR << " or " << COMB_MAX << ")." << std::endl;
    std::cerr << "    BOOL is set to 'true' if the rnn_path refers to an error-correcting RNN." << std::endl;
    std::cerr << "(Note: --save_path and --comb_type are only required if both --ngram_path and --rnn_path are set.)" << std::endl;
}

int main(int argc, char* argv[]) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    std::string ngram_path;
    std::string rnn_path;
    std::string test_input_path;
    std::string test_target_path;
    std::string save_path;
    std::string comb_type;
    std::string error_rnn;

    const bool parse_result = tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("ngram_path", &ngram_path),
        tensorflow::Flag("rnn_path", &rnn_path),
        tensorflow::Flag("test_input_path", &test_input_path),
        tensorflow::Flag("test_target_path", &test_target_path),
        tensorflow::Flag("save_path", &save_path),
        tensorflow::Flag("comb_type", &comb_type),
        tensorflow::Flag("error_rnn", &error_rnn),
    });
    if (!parse_result) {
        usage(argv[0]);
        return -1;
    }

    if (ngram_path.empty() && rnn_path.empty()) {
        std::cerr << "Error: at least one of --rnn_path or --ngram_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (!ngram_path.empty() && !rnn_path.empty()) {
        if (save_path.empty()) {
            std::cerr << "Error: --save_path must be set if both --ngram_path and --rnn_path are set." << std::endl;
            usage(argv[0]);
            return -1;
        }
        if (comb_type.empty()) {
            std::cerr << "Error: --comb_type must be set if both --ngram_path and --rnn_path are set." << std::endl;
            usage(argv[0]);
            return -1;
        }
    }
    if (test_input_path.empty()) {
        std::cerr << "Error: --test_input_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (test_target_path.empty()) {
        test_target_path = test_input_path;
    }

    LM *lm;
    NGram *ngram_lm;
    RNN *rnn_lm;
    if (!ngram_path.empty()) {
        ngram_lm = Load(ngram_path);
    }
    if (!rnn_path.empty()) {
        if (error_rnn.empty()) {
            rnn_lm = new RNN(rnn_path);
        } else {
            rnn_lm = new ErrorCorrectingRNN(rnn_path, "/Users/devankuleindiren/Downloads/words.txt");
        }
    }

    if (!ngram_path.empty()) {
        if (!rnn_path.empty()) {
            if (comb_type.compare(COMB_AVR) == 0) {
                lm = new NGramRNNAverage(ngram_lm, rnn_lm);
            } else if (comb_type.compare(COMB_MAX) == 0) {
                lm = new NGramRNNMax(ngram_lm, rnn_lm);
            } else {
                std::cerr << "Error: " << comb_type << " is not a valid --comb_type." << std::endl;
                usage(argv[0]);
                return -1;
            }
        } else {
            lm = ngram_lm;
        }
    } else {
        lm = rnn_lm;
    }

    Benchmark *benchmark = new Benchmark(lm);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    std::cout << "Calculating perplexity..." << std::endl;
    double perplexity = benchmark->Perplexity(test_input_path, test_target_path, false);
    std::cout << "Perplexity = " << perplexity << std::endl;
    std::cout << "Completed in ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    std::cout << "Calculating average keys saved..." << std::endl;
    double average_keys_saved = benchmark->AverageKeysSaved(test_input_path, test_target_path, 1000);
    std::cout << "Average keys saved = " << average_keys_saved << std::endl;
    std::cout << "Completed in ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    std::cout << "Calculating guessing entropy..." << std::endl;
    double guessing_entropy = benchmark->GuessingEntropy(test_input_path, test_target_path, 1000);
    std::cout << "Guessing entropy = " << guessing_entropy << std::endl;
    std::cout << "Completed in ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << " seconds." << std::endl;

    start = std::chrono::steady_clock::now();
    std::cout << "Calculating average inference time..." << std::endl;
    long average_inference_time_us = benchmark->AverageInferenceTimeMicroSeconds(test_input_path, 50);
    std::cout << "Average inference time = " << average_inference_time_us << "us" << std::endl;
    std::cout << "Completed in ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << " seconds." << std::endl;

    std::cout << "Calculating physical memory usage..." << std::endl;
    long physical_memory_usage_bytes = benchmark->PhysicalMemoryUsageBytes();
    std::cout << "Physical memory usage = " << physical_memory_usage_bytes << "bytes" << std::endl;


    tensorflow::Source::benchmark::BenchmarkProto benchmark_proto;
    benchmark_proto.set_perplexity(perplexity);
    benchmark_proto.set_average_keys_saved(average_keys_saved);
    benchmark_proto.set_guessing_entropy(guessing_entropy);
    benchmark_proto.set_average_inference_time_us(average_inference_time_us);
    benchmark_proto.set_physical_memory_usage_bytes(physical_memory_usage_bytes);

    std::string benchmark_dir_path;
    if (!save_path.empty()) {
        benchmark_dir_path = save_path;
    } else if (!ngram_path.empty()) {
        benchmark_dir_path = ngram_path;
    } else {
        benchmark_dir_path = rnn_path;
    }
    if (benchmark_dir_path.back() != '/') {
        benchmark_dir_path += '/';
    }

    std::ofstream ofs (benchmark_dir_path + "benchmark.pbtxt", std::ios::out | std::ios::trunc);
    google::protobuf::io::OstreamOutputStream osos(&ofs);
    if (!google::protobuf::TextFormat::Print(benchmark_proto, &osos)) {
        std::cerr << "Failed to write benchmark proto." << std::endl;
    } else {
        std::cout << "Saved benchmark proto." << std::endl;
    }
}
