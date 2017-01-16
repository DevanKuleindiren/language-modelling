# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Example / benchmark for building a PTB LSTM model.
Trains the model described in:
(Zaremba, et. al.) Recurrent Neural Network Regularization
http://arxiv.org/abs/1409.2329
There are 3 supported model configurations:
===========================================
| config | epochs | train | valid  | test
===========================================
| small  | 13     | 37.99 | 121.39 | 115.91
| medium | 39     | 48.45 |  86.16 |  82.07
| large  | 55     | 37.87 |  82.62 |  78.29
The exact results may vary depending on the random initialization.
The hyperparameters used in the model:
- init_scale - the initial scale of the weights
- learning_rate - the initial value of the learning rate
- max_grad_norm - the maximum permissible norm of the gradient
- num_layers - the number of LSTM layers
- num_steps - the number of unrolled steps of LSTM
- hidden_size - the number of LSTM units
- max_epoch - the number of epochs trained with the initial learning rate
- max_max_epoch - the total number of epochs for training
- keep_prob - the probability of keeping weights in the dropout layer
- lr_decay - the decay of the learning rate for each epoch after "max_epoch"
- batch_size - the batch size
The data required for this example is in the data/ dir of the
PTB dataset from Tomas Mikolov's webpage:
$ wget http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
$ tar xvf simple-examples.tgz
To run:
$ python ptb_word_lm.py --data_path=simple-examples/data/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy as np
import tensorflow as tf

from tensorflow.Source.lm.rnn import ptb_reader

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("model_path", None, "The path to the model file.")

FLAGS = flags.FLAGS

class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 2
  num_steps = 20
  hidden_size = 200
  max_epoch = 4
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 10001

def run_epoch(session, data, config):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0

  logits = tf.get_default_graph().get_tensor_by_name("inference/lstm/logits:0")

  for step, (x, y) in enumerate(ptb_reader.ptb_iterator(data, config.batch_size,
                                                    config.num_steps)):
    feed_dict = {}
    feed_dict = {"inference/lstm/inputs:0": x}
    loss = tf.nn.seq2seq.sequence_loss_by_example(
        [logits],
        [tf.reshape(y, [-1])],
        [tf.ones([config.batch_size * config.num_steps], dtype=tf.float32)])
    cost = tf.reduce_sum(loss) / config.batch_size

    fetches = [cost]

    cost_f, = session.run(fetches, feed_dict)

    costs += cost_f
    iters += config.num_steps

    if step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * config.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)


def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")

  test_data = ptb_reader.ptb_raw_data(FLAGS.data_path)

  config = SmallConfig()
  config.batch_size = 1

  graph_def = tf.GraphDef()
  with open(FLAGS.model_path + "/graph.pb", "rb") as f:
      graph_def.ParseFromString(f.read())
  tf.import_graph_def(graph_def, name="")

  with tf.Session() as session:

    tf.initialize_all_variables().run()

    test_perplexity = run_epoch(session, test_data, config)
    print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
