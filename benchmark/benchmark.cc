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
            std::list<std::string> ngram;

            for (int i = 0; i < 5; i++) {
                ngram.push_back("<s>");
            }

            while (!line.empty()) {
                pos = line.find(" ");
                if (pos == std::string::npos) {
                    pos = line.size();
                }

                std::string word = line.substr(0, pos);
                ngram.push_back(word);
                num_words++;

                if (language_model->ContainsWord(word)) {
                    double fraction = 1 / language_model->Prob(ngram);
                    double new_product = product * fraction;
                    if (isinf(new_product)) {
                        products.push_back(product);
                        product = fraction;
                    } else {
                        product = new_product;
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
