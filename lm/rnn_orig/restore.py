from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.Source.lm.rnn_orig import reader

flags = tf.flags

flags.DEFINE_string("model_path", None, "The directory path to the model file.")
flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("save_path", None, "The directory to store the model & checkpoints.")

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
  vocab_size = 10000

def run_epoch(session, config, data, verbose=True):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // config.batch_size) - 1) // config.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state_fetches = {
    "c_0": "test/lstm/zeros:0",
    "h_0": "test/lstm/zeros_1:0",
    "c_1": "test/lstm/zeros_2:0",
    "h_1": "test/lstm/zeros_3:0",
  }
  state = session.run(state_fetches)
  for step, (x, y) in enumerate(reader.ptb_iterator(data, config.batch_size,
                                                    config.num_steps)):
    fetches = {
      "cost": "test/lstm/reduce_sum:0",
      "f_c_0": "test/lstm/RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2:0",
      "f_h_0": "test/lstm/RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2:0",
      "f_c_1": "test/lstm/RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2:0",
      "f_h_1": "test/lstm/RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2:0",
    }
    feed_dict = {
      "test/lstm/inputs:0": x,
      "test/lstm/targets:0": y,
      "test/lstm/zeros:0": state["c_0"],
      "test/lstm/zeros_1:0": state["h_0"],
      "test/lstm/zeros_2:0": state["c_1"],
      "test/lstm/zeros_3:0": state["h_1"],
    }

    vals = session.run(fetches, feed_dict)

    cost = vals["cost"]
    state["c_0"] = vals["f_c_0"]
    state["h_0"] = vals["f_h_0"]
    state["c_1"] = vals["f_c_1"]
    state["h_1"] = vals["f_h_1"]

    costs += cost
    iters += config.num_steps

    if verbose and step % (epoch_size // 100) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * config.batch_size / (time.time() - start_time)))

  return np.exp(costs / iters)

def main(_):
  if not FLAGS.data_path:
    raise ValueError("Must set --data_path to PTB data directory")
  if not FLAGS.save_path:
    raise ValueError("Must set --save_path to PTB model directory")
  if not FLAGS.model_path:
    raise ValueError("Must set --model_path to file path of checkpoint.")

  raw_data = reader.ptb_raw_data(FLAGS.data_path)
  train_data, valid_data, test_data, _ = raw_data

  eval_config = SmallConfig()
  eval_config.batch_size = 1
  eval_config.num_steps = 1

  with tf.Session() as session:
    #saver = tf.train.import_meta_graph(FLAGS.checkpoint_path + ".meta")
    #saver.restore(session, FLAGS.checkpoint_path)
    graph_def = tf.GraphDef()
    with open(os.path.join(FLAGS.model_path, "graph.pb"), "rb") as f:
      graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name="")

    test_perplexity = run_epoch(session, eval_config, test_data)
    print("Test Perplexity: %.3f" % test_perplexity)

    # Save perplexities.
    perplexity_path = os.path.join(FLAGS.save_path, "perplexity.txt")
    with open(perplexity_path, 'w+') as f:
      f.write("Test perplexity = %f\n" % test_perplexity)


if __name__ == "__main__":
  tf.app.run()
