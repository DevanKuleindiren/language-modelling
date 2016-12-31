#include "benchmark.h"

double Benchmark::Perplexity(std::string file_name) {
    std::ifstream f (file_name);

    if (f.is_open()) {
        std::string line;
        double product = 1;
        double num_words = 0;
        std::list<double> products;

        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;
            std::list<std::string> seq;
            seq.push_back("<s>");

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }

                if (pos > 0) {
                    std::string word = line.substr(0, pos);
                    seq.push_back(word);
                    num_words++;

                    if (language_model->ContainsWord(word)) {
                        double fraction = 1 / language_model->Prob(seq);
                        double new_product = product * fraction;
                        if (isinf(new_product)) {
                            products.push_back(product);
                            product = fraction;
                        } else {
                            product = new_product;
                        }
                    }
                }
                line.erase(0, pos + 1);
            }
        }

        products.push_back(product);
        double perplexity = 1.0;
        for (std::list<double>::iterator it = products.begin(); it != products.end(); ++it) {
            perplexity *= pow(*it, 1 / num_words);
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

        while (std::getline(f, line)) {
            size_t pos = 0;
            std::string word;
            std::list<std::string> seq;
            seq.push_back("<s>");

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }

                if (pos > 0) {
                    std::string word = line.substr(0, pos);
                    seq.push_back(word);
                    num_words++;

                    if (language_model->ContainsWord(word)) {
                        double logp = log(language_model->Prob(seq));
                        double new_sum = sum + logp;
                        if (isinf(new_sum)) {
                            sums.push_back(sum);
                            sum = logp;
                        } else {
                            sum = new_sum;
                        }
                    }
                }
                line.erase(0, pos + 1);
            }
        }

        sums.push_back(sum);
        double perplexity = 1.0;
        for (std::list<double>::iterator it = sums.begin(); it != sums.end(); ++it) {
            perplexity *= exp((-1.0 / num_words) * *it);
        }
        return perplexity;
    }
    return 0;
}
