#include <time.h>
#include <vector>
#include "ngram.h"
#include "smoothing/absolute_discounting.h"
#include "smoothing/add_one.h"
#include "smoothing/katz.h"
#include "smoothing/kneser_ney_mod.h"
#include "smoothing/kneser_ney.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/util/command_line_flags.h"

#define ABSD std::string("absolute_discounting")
#define ADD1 std::string("add_one")
#define KATZ std::string("katz")
#define KNES std::string("kneser_ney")
#define KNSM std::string("kneser_ney_mod")

void usage(char* const argv_0) {
    std::cerr << "Usage: " << argv_0
              << " --n=N --min_frequency=MIN_FREQ --smoothing=SMOOTH --training_path=T_PATH "
              << "--save_path=S_PATH" << std::endl;
    std::cerr << "Where:" << std::endl;
    std::cerr << "    N        is the N (a positive integer) in N-gram." << std::endl;
    std::cerr << "    MIN_FREQ is the minimum number of times a word must be seen to not be OOV." << std::endl;
    std::cerr << "    SMOOTH   is the smoothing method applied (one of: ";
    std::cerr << ABSD << ", " << ADD1 << ", " << KATZ << ", " << KNES  << " or " << KNSM << ")." << std::endl;
    std::cerr << "    T_PATH   is the file path to the training data." << std::endl;
    std::cerr << "    S_PATH   is the directory in which the trained model should be saved." << std::endl;
}

int main(int argc, char* argv[]) {
    tensorflow::port::InitMain(argv[0], &argc, &argv);

    int n = 0;
    int min_frequency = 1;
    tensorflow::string smoothing;
    std::string training_path;
    std::string save_path;

    const bool parse_result = tensorflow::ParseFlags(&argc, argv, {
        tensorflow::Flag("n", &n),
        tensorflow::Flag("min_frequency", &min_frequency),
        tensorflow::Flag("smoothing", &smoothing),
        tensorflow::Flag("training_path", &training_path),
        tensorflow::Flag("save_path", &save_path),
    });
    if (!parse_result) {
        usage(argv[0]);
        return -1;
    }

    if (n == 0) {
        std::cerr << "Error: --n must be set and given a positive integer value." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (training_path.empty()) {
        std::cerr << "Error: --training_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }
    if (save_path.empty()) {
        std::cerr << "Error: --save_path must be set." << std::endl;
        usage(argv[0]);
        return -1;
    }

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    NGram *lm;
    if (smoothing.compare(ABSD) == 0) {
        lm = new AbsoluteDiscounting(training_path, n, min_frequency);
    } else if (smoothing.compare(ADD1) == 0) {
        lm = new AddOne(training_path, n, min_frequency);
    } else if (smoothing.compare(KATZ) == 0) {
        lm = new Katz(training_path, n, min_frequency);
    } else if (smoothing.compare(KNES) == 0) {
        lm = new KneserNey(training_path, n, min_frequency);
    } else if (smoothing.compare(KNSM) == 0) {
        lm = new KneserNeyMod(training_path, n, min_frequency);
    } else {
        lm = new NGram(training_path, n, min_frequency);
    }
    std::cout << "Completed in ";
    std::cout << std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start).count();
    std::cout << " seconds." << std::endl;
    lm->Save(save_path);
}
