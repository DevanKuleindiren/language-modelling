#include "benchmark.h"

double Benchmark::Perplexity(std::string input_file_name, std::string target_file_name, bool use_exp_calculation) {
    DualFileReader *dual_file_reader = PrepareDualReader(input_file_name, target_file_name);

    // If using the product method, then x represents a product, otherwise, it represents a sum.
    double batch = use_exp_calculation ? 0 : 1;
    int num_words = 0;
    std::list<double> batches;
    std::list<std::string> seq;
    std::string input_word;
    std::string target_word;

    while (dual_file_reader->GetNextInputTargetWordPair(&input_word, &target_word)) {
        seq.push_back(target_word);

        while (seq.size() > language_model->ContextSize().second) {
            seq.pop_front();
        }

        double prob = language_model->Prob(seq);
        if (prob <= 0) {
            // To avoid division by 0, we instead set the probability to be a small value. It means that the
            // perplexity for language models that run into this case aren't mathematically exact, but it does give
            // results that are easier to compare.
            prob = 1e-9;
        }

        // Swap the target word for the input word now that the probability has been estimated.
        seq.pop_back();
        seq.push_back(input_word);

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

double Benchmark::AverageKeysSaved(std::string input_file_name, std::string target_file_name, int max_words) {
    // The average number of characters saved (per word), where 'saved' characters are those characters of the word
    // that don't need to be typed as a result of the correct word being predicted in the top 3.

    DualFileReader *dual_file_reader = PrepareDualReader(input_file_name, target_file_name);

    int num_words = 0;
    double num_chars = 0;
    double keys_saved = 0;
    std::list<std::string> seq;
    std::string input_word;
    std::string target_word;

    CharTrie *char_trie = new CharTrie();
    for (std::unordered_map<std::string, size_t>::const_iterator it = language_model->GetVocab()->begin(); it != language_model->GetVocab()->end(); ++it) {
        char_trie->Insert(it->first, 0);
    }

    RNN *rnn;
    bool use_rnn = false;
    if ((rnn = dynamic_cast<RNN*>(language_model))) {
        use_rnn = true;
    }

    while (dual_file_reader->GetNextInputTargetWordPair(&input_word, &target_word) && num_words < max_words) {
        while (seq.size() >= language_model->ContextSize().second) {
            seq.pop_front();
        }

        if (seq.size() >= (language_model->ContextSize()).first) {
            num_words++;
            num_chars += target_word.size();

            // Avoid evaluating the softmax layer in RNN models.
            if (use_rnn) {
                rnn->LogitsAllFollowing(seq, char_trie);
            } else {
                language_model->ProbAllFollowing(seq, char_trie);
            }

            bool next_word_predicted = false;
            for (int i = 0; i <= target_word.length() && !next_word_predicted; ++i) {
                std::list<std::pair<std::string, double>> top_3 = char_trie->GetMaxKWithPrefix(target_word.substr(0, i), 3);

                for (std::list<std::pair<std::string, double>>::iterator it = top_3.begin(); it != top_3.end(); ++it) {
                    if (target_word.compare(it->first) == 0) {
                        keys_saved += target_word.length() - i;
                        next_word_predicted = true;
                        break;
                    }
                }
            }

            if (num_words > 0 && num_words % 100 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
        seq.push_back(input_word);
    }

    if (num_chars > 0) {
        return (keys_saved / num_chars);
    } else {
        return 0;
    }
}

double Benchmark::GuessingEntropy(std::string input_file_name, std::string target_file_name, int max_words) {
    DualFileReader *dual_file_reader = PrepareDualReader(input_file_name, target_file_name);

    int num_words = 0;
    double total_entropy = 0;
    std::list<std::string> seq;
    std::string input_word;
    std::string target_word;

    RNN *rnn;
    bool use_rnn = false;
    if ((rnn = dynamic_cast<RNN*>(language_model))) {
        use_rnn = true;
    }

    while (dual_file_reader->GetNextInputTargetWordPair(&input_word, &target_word) && num_words < max_words) {
        while (seq.size() >= language_model->ContextSize().second) {
            seq.pop_front();
        }

        if (seq.size() >= (language_model->ContextSize()).first) {
            num_words++;

            std::list<std::pair<std::string, double>> probs;
            // Avoid evaluating the softmax layer in RNN models.
            if (use_rnn) {
                rnn->LogitsAllFollowing(seq, probs);
            } else {
                language_model->ProbAllFollowing(seq, probs);
            }

            double prob_of_target;
            for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                if (target_word.compare(it->first) == 0) {
                    prob_of_target = it->second;
                }
            }

            // Count the number of words with a higher probability than to_predict.
            int num_higher = 0;
            for (std::list<std::pair<std::string, double>>::iterator it = probs.begin(); it != probs.end(); ++it) {
                if (it->second > prob_of_target) {
                    num_higher++;
                }
            }

            total_entropy += log2(num_higher + 1);

            if (num_words > 0 && num_words % 100 == 0) {
                std::cout << "Processed " << num_words << " words." << std::endl;
            }
        }
        seq.push_back(input_word);
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
    FileReader *file_reader = new FileReader(file_name);
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

            if (num_words > 0 && num_words % 100 == 0) {
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

DualFileReader *Benchmark::PrepareDualReader(std::string input_file_name, std::string target_file_name) {
    if (input_file_name.compare(target_file_name) == 0) {
        return new DuplicatedDualFileReader(input_file_name);
    } else {
        return new DifferentDualFileReader(input_file_name, target_file_name);
    }
}
