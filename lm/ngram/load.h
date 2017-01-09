#ifndef load_h
#define load_h

#include <fstream>
#include <iostream>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <string>
#include "ngram.h"
#include "prob_trie.h"
#include "tensorflow/Source/lm/ngram/ngram.pb.h"
#include "tensorflow/Source/lm/ngram/smoothing/absolute_discounting.h"
#include "tensorflow/Source/lm/ngram/smoothing/add_one.h"
#include "tensorflow/Source/lm/ngram/smoothing/katz.h"
#include "tensorflow/Source/lm/ngram/smoothing/kneser_ney_mod.h"
#include "tensorflow/Source/lm/ngram/smoothing/kneser_ney.h"
#include "tensorflow/Source/lm/vocab.h"

NGram *Load(std::string directory_path);

#endif // load.h
