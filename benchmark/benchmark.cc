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

double Benchmark::AverageKeysSaved(std::string file_name, int max_words) {
    // The average number of characters saved (per word), where 'saved' characters are those characters of the word
    // that don't need to be typed as a result of the correct word being predicted in the top 3.

    std::ifstream file_stream (file_name);
    FileReader *file_reader = new FileReader(file_stream);
    int num_words = 0;
    double keys_saved = 0;
    std::list<std::string> seq;
    std::string word;

    // Only populate once, and then afterwards only call Update() for efficiency.
    CharTrie *char_trie = new CharTrie();
    bool char_trie_populated = false;

    while (file_reader->GetNextWord(&word) && num_words < max_words) {
        seq.push_back(word);

        while (seq.size() > language_model->ContextSize().second) {
            seq.pop_front();
        }

        if (seq.size() > (language_model->ContextSize()).first) {
            num_words++;

            std::string to_predict = seq.back();
            seq.pop_back();
            std::list<std::pair<std::string, double>> probs;
            language_model->ProbAllFollowing(seq, probs);
            seq.push_back(to_predict);

            if (char_trie_populated) {
                for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                    char_trie->Update(it->first, it->second);
                }
            } else {
                for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                    char_trie->Insert(it->first, it->second);
                }
            }

            bool next_word_predicted = false;
            for (int i = 0; i <= to_predict.length() && !next_word_predicted; ++i) {
                std::list<std::pair<std::string, double>> top_3 = char_trie->GetMaxKWithPrefix(to_predict.substr(0, i), 3);

                for (std::list<std::pair<std::string, double>>::iterator it = top_3.begin(); it != top_3.end(); ++it) {
                    if (to_predict.compare(it->first) == 0) {
                        keys_saved += to_predict.length() - i;
                        next_word_predicted = true;
                        break;
                    }
                }
            }

            if (num_words > 0 && num_words % 1000 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
    }

    if (num_words > 0) {
        return (keys_saved / num_words);
    } else {
        return 0;
    }
}

double Benchmark::GuessingEntropy(std::string file_name, int max_words) {
    std::ifstream file_stream (file_name);
    FileReader *file_reader = new FileReader(file_stream);
    int num_words = 0;
    double total_entropy = 0;
    std::list<std::string> seq;
    std::string word;

    while (file_reader->GetNextWord(&word) && num_words < max_words) {
        seq.push_back(word);

        while (seq.size() > language_model->ContextSize().second) {
            seq.pop_front();
        }

        if (seq.size() > (language_model->ContextSize()).first) {
            num_words++;

            std::string to_predict = seq.back();
            seq.pop_back();
            std::list<std::pair<std::string, double>> probs;
            language_model->ProbAllFollowing(seq, probs);
            seq.push_back(to_predict);

            double prob_of_to_predict;
            for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                if (to_predict.compare(it->first) == 0) {
                    prob_of_to_predict = it->second;
                }
            }

            // Count the number of words with a higher probability than to_predict.
            int num_higher = 0;
            for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                if (it->second > prob_of_to_predict) {
                    num_higher++;
                }
            }

            total_entropy += log2(num_higher + 1);

            if (num_words > 0 && num_words % 1000 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
    }

    if (num_words > 0) {
        return (total_entropy / num_words);
    } else {
        return 0;
    }
}

long Benchmark::PhysicalMemoryUsageBytes() {
    struct task_basic_info t_info;
    mach_msg_type_number_t t_info_count = TASK_BASIC_INFO_COUNT;
    if (KERN_SUCCESS != task_info(mach_task_self(), TASK_BASIC_INFO, (task_info_t)&t_info, &t_info_count)) {
        return -1;
    }
    return t_info.resident_size;
}

long Benchmark::AverageInferenceTimeMicroSeconds(std::string file_name, int max_words) {
    std::ifstream file_stream (file_name);
    FileReader *file_reader = new FileReader(file_stream);
    int num_words = 0;
    std::list<std::string> seq;
    std::string word;
    long total_duration = 0;

    while (file_reader->GetNextWord(&word) && num_words < max_words) {
        seq.push_back(word);

        while (seq.size() >= language_model->ContextSize().second) {
            seq.pop_front();
        }

        if (seq.size() >= (language_model->ContextSize()).first) {
            num_words++;

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            std::list<std::pair<std::string, double>> probs;
            language_model->ProbAllFollowing(seq, probs);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            total_duration += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            if (num_words > 0 && num_words % 1000 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
    }

    if (num_words > 0) {
        return (total_duration / max_words);
    } else {
        return 0;
    }
}
