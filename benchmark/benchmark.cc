#include "benchmark.h"

double Benchmark::Perplexity(std::string file_name, bool use_exp_calculation) {
    std::ifstream file_stream (file_name);
    FileReader *file_reader = new FileReader(file_stream);
    int num_words = 0;

    // If using the product method, then x represents a product, otherwise, it represents a sum.
    double batch = use_exp_calculation ? 0 : 1;
    std::list<double> batches;
    std::list<std::string> seq;
    std::string word;

    while (file_reader->GetNextWord(&word)) {
        seq.push_back(word);

        while (seq.size() > language_model->ContextSize().second) {
            seq.pop_front();
        }

        double prob;
        if (seq.size() > (language_model->ContextSize()).first &&
            language_model->ContainsWord(word) &&
            (prob = language_model->Prob(seq)) > 0) {

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

            if (num_words > 0 && num_words % 1000 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
    }

    batches.push_back(batch);
    double perplexity = 1.0;
    for (std::list<double>::iterator it = batches.begin(); it != batches.end(); ++it) {
        if (use_exp_calculation) {
            perplexity *= exp((-1.0 / ((double) num_words)) * *it);
        } else {
            perplexity *= pow(*it, 1 / ((double) num_words));
        }
    }
    return perplexity;
}
