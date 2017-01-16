#include "benchmark.h"

double Benchmark::Perplexity(std::string file_name, bool use_exp_calculation) {
    std::ifstream f (file_name);

    if (f.is_open()) {
        std::string line;
        double num_words = 0;
        int num_skipped = 0;

        // If using the product method, then x represents a product, otherwise, it represents a sum.
        double batch = use_exp_calculation ? 0 : 1;
        std::list<double> batches;

        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;
            std::list<std::string> seq;
            for (int i = 0; i < (language_model->ContextSize()).first; i++) {
                seq.push_back("<s>");
            }

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }

                if (pos > 0) {
                    std::string word = line.substr(0, pos);
                    seq.push_back(word);

                    double prob = language_model->Prob(seq);
                    if (language_model->ContainsWord(word) && prob > 0) {
                        num_words++;
                        double tmp;
                        double new_batch;
                        if (use_exp_calculation) {
                            tmp = log(prob);
                            new_batch = batch + tmp;
                        } else {
                            tmp = 1 / prob;
                            new_batch = batch * tmp;
                        }
                        if (isinf(new_batch)) {
                            batches.push_back(batch);
                            batch = tmp;
                        } else {
                            batch = new_batch;
                        }
                    } else {
                        num_skipped++;
                    }
                }
                line.erase(0, pos + 1);

                if (num_words > 0 && ((int) num_words) % 1000 == 0) {
                    std::cout << "Processed " << num_words << " words." << std::endl;
                }
            }
        }

        batches.push_back(batch);
        double perplexity = 1.0;
        for (std::list<double>::iterator it = batches.begin(); it != batches.end(); ++it) {
            if (use_exp_calculation) {
                perplexity *= exp((-1.0 / num_words) * *it);
            } else {
                perplexity *= pow(*it, 1 / num_words);
            }
        }

        if (num_skipped > 0) {
            std::cout << "(Skipped " << num_skipped << " cases where the probability was 0)." << std::endl;
        }

        return perplexity;
    }
    return 0;
}
