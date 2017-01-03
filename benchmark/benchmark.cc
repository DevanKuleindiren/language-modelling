#include "benchmark.h"

double Benchmark::Perplexity(std::string file_name) {
    std::ifstream f (file_name);

    if (f.is_open()) {
        std::string line;
        double product = 1;
        double num_words = 0;
        std::list<double> products;
        bool did_skip_zero_prob = false;

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
                    if (prob > 0) {
                        num_words++;
                        double fraction = 1 / prob;
                        double new_product = product * fraction;
                        if (isinf(new_product)) {
                            products.push_back(product);
                            product = fraction;
                        } else {
                            product = new_product;
                        }
                    } else {
                        did_skip_zero_prob = true;
                    }
                }
                line.erase(0, pos + 1);

                if (((int) num_words) % 1000 == 0) {
                    std::cout << "Processed " << num_words << " words." << std::endl;
                }
            }
        }

        products.push_back(product);
        double perplexity = 1.0;
        for (std::list<double>::iterator it = products.begin(); it != products.end(); ++it) {
            perplexity *= pow(*it, 1 / num_words);
        }

        if (did_skip_zero_prob) {
            std::cout << "(skipped at least one P=0) ";
        }

        return perplexity;
    }
    return 0;
}

double Benchmark::PerplexityExp(std::string file_name) {
    std::ifstream f (file_name);

    if (f.is_open()) {
        std::string line;
        double sum = 0;
        double num_words = 0;
        std::list<double> sums;
        bool did_skip_zero_prob = false;

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
                    if (prob > 0) {
                        num_words++;
                        double logp = log(prob);
                        double new_sum = sum + logp;
                        if (isinf(new_sum)) {
                            sums.push_back(sum);
                            sum = logp;
                        } else {
                            sum = new_sum;
                        }
                    } else {
                        did_skip_zero_prob = true;
                    }
                }
                line.erase(0, pos + 1);

                if (((int) num_words) % 1000 == 0) {
                    std::cout << "Processed " << num_words << " words." << std::endl;
                }
            }
        }

        sums.push_back(sum);
        double perplexity = 1.0;
        for (std::list<double>::iterator it = sums.begin(); it != sums.end(); ++it) {
            perplexity *= exp((-1.0 / num_words) * *it);
        }

        if (did_skip_zero_prob) {
            std::cout << "(skipped at least one P=0) ";
        }

        return perplexity;
    }
    return 0;
}
