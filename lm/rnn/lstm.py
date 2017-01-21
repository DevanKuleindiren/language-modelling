import heapq
import reader
import numpy as np
import os
import tensorflow as tf
import time

from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.Source.lm import vocab_pb2

tf.flags.DEFINE_bool("infer", False, "Run inference on a previously saved model.")
tf.flags.DEFINE_string("training_data_path", None, "The path to the training data.")
tf.flags.DEFINE_string("save_path", None, "The path to save the model.")
tf.flags.DEFINE_string("size", None, "The size of the lstm model (one of: small, large).")

FLAGS = tf.flags.FLAGS

class SmallConfig:
    """
    The hyperparameters used in the model:
    - batch_size - the batch size
    - hidden_size - the number of LSTM units
    - init_scale - the initial scale of the weights
    - keep_prob - the probability of keeping weights in the dropout layer
    - lr - the initial value of the learning rate
    - lr_decay - the decay of the learning rate for each epoch after "max_epoch"
    - max_grad_norm - the maximum permissible norm of the gradient
    - max_epoch - the number of epochs trained with the initial learning rate
    - max_max_epoch - the total number of epochs for training
    - min_frequency - the minimum number of times a word needs to be seen to be considered part of the vocabulary
    - num_layers - the number of LSTM layers
    - num_steps - the number of unrolled steps of LSTM
    """
    batch_size = 20
    hidden_size = 200
    init_scale = 0.1
    keep_prob = 1.0
    lr = 1.0
    lr_decay = 0.5
    max_epoch = 4
    max_grad_norm = 5
    max_max_epoch = 13
    min_frequency = 1
    num_layers = 2
    num_steps = 20

class LargeConfig:
    """
    The hyperparameters used in the model are as specified above.
    """
    batch_size = 20
    hidden_size = 1500
    init_scale = 0.04
    keep_prob = 0.35
    lr = 1.0
    lr_decay = 1 / 1.15
    max_epoch = 14
    max_grad_norm = 10
    max_max_epoch = 55
    min_frequency = 1
    num_layers = 2
    num_steps = 35

class LSTM:

    def __init__(self, config, vocab_size, epoch_size, is_training):

        self._config = config
        self._epoch_size = epoch_size
        self._vocab_size = vocab_size

        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name="inputs")
        self._target_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name="targets")

        # A 'cell' in TensorFlow actually refers to an array of the LSTMs cells described in literature, so this is an
        # array of config.hidden_size LSTM cells.
        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)

        # This adds dropout to the LSTM cells by applying the probability of keeping a weight.
        if is_training and config.keep_prob < 1:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)

        # This adds layers to the RNN to give config.num_layers layers overall.
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)

        # Initialise the LSTM cell weights.
        self._initial_state = cell.zero_state(config.batch_size, dtype=tf.float32)

        with tf.device("/cpu:0"):
            # Create a (vocab_size x config.hidden_size) size embedding matrix.
            embedding = tf.get_variable("embedding", [vocab_size, config.hidden_size], dtype=tf.float32)

            # This converts the (config.batch_size x config.num_steps) input Tensor to a
            # (config.batch_size x config.num_steps x config.hidden_size) Tensor by replacing each word id, X, with the
            # Xth row of the embedding matrix.
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # Apply dropout.
        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)


        outputs = []
        state = self._initial_state
        with tf.variable_scope("LSTM"):
            for time_step in range(config.num_steps):
                # We want to use the same set of weights in each time step. This is due to the recurrent structure of
                # RNNs.
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                # Each LSTM cell has a __call__ function which takes a batch of words at a particular time step
                # (represented as embedding vectors) and returns the tuple (h, c) where h is the new
                # LSTM activation and c is the new cell state as shown in Figure 1 of http://arxiv.org/abs/1409.2329.
                #
                # For example, if we denote {cat} as the embedding vector for the word "cat", then we might have
                # something like:
                #
                # inputs = [[{the}, {cat}, {sat}, {on}],
                #           [{walk}, {in}, {the}, {park}],
                #           ...
                #           [{grab}, {a}, {spot}, {by}]]
                #
                # and then at time step 2, we have:
                #
                # inputs[:, 2, :] = [{cat},
                #                    {in},
                #                    ...
                #                    {a}]
                #
                # The shape of cell_output is (config.batch_size x config.hidden_size).
                (cell_output, state) = cell(inputs[:, time_step, :], state)

                # Each LSTM cell activation is stored in a list of outputs.
                outputs.append(cell_output)

        # Concatenate the outputs from each time step along dimension 1 (which should give a matrix of shape
        # ((config.batch_size * config.num_steps) x config.hidden_size).
        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])

        softmax_w = tf.get_variable(
            "softmax_w", [config.hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # Multiply the LSTM activations and add bias to give logits of shape
        # (config.batch_size * config.num_steps) x vocab_size. Note that tf.add() doesn't require the Tensor shapes to
        # match due to broadcasting.
        logits = tf.add(tf.matmul(output, softmax_w), softmax_b, name="logits")
        self._logits = logits

        # When running inference it is useful to normalise the outputs so that they represent probabilities using the
        # softmax function.
        self._predictions = tf.nn.softmax(logits, name="predictions")

        # The cross-entropy loss is calculated between the logits and the targets (which are flattened into a Tensor
        # of shape (config.batch_size * config.num_steps). The tf.ones() are just weights in the weighted cross-entropy
        # loss.
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._target_data, [-1])],
            [tf.ones([config.batch_size * config.num_steps], dtype=tf.float32)])

        # The total loss is divided by config.batch_size which gives an average cost per example.
        self._cost = cost = tf.reduce_sum(loss) / config.batch_size
        self._final_state = state

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                          config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def target_data(self):
        return self._target_data

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def logits(self):
        return self._logits

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op

    @property
    def epoch_size(self):
        return self._epoch_size

    @property
    def predictions(self):
        return self._predictions


def run_epoch(sess, model, input_data):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "train_op": model.train_op,
    }

    for step, (x, t) in enumerate(reader.batch_producer(input_data, model.config.batch_size, model.config.num_steps)):
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.target_data] = t

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.config.num_steps

        if step % 10 == 0:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.epoch_size, np.exp(costs / iters),
                 iters * model.config.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)

def predict(sess, inputs, id_to_word, seq_len):
    fetches = {"predictions": "inference/lstm/predictions:0"}
    feed_dict = {"inference/lstm/inputs:0": inputs}
    vocab_size = len(id_to_word)

    vals = sess.run(fetches, feed_dict)
    predictions = vals["predictions"]

    return sorted(heapq.nlargest(10, [(id_to_word[i], p) for (i, p) in zip(xrange(vocab_size), predictions[seq_len - 1])], key=lambda x: x[1]), key=lambda x: -x[1])

def main(_):
    if not FLAGS.training_data_path:
        raise ValueError("Must set --training_data_path.")
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path.")
    if not FLAGS.size:
        raise ValueError("Must set --size.")

    with tf.Graph().as_default():
        if FLAGS.size == "small":
            train_config = SmallConfig()
            infer_config = SmallConfig()
        elif FLAGS.size == "large":
            train_config = LargeConfig()
            infer_config = LargeConfig()
        else:
            raise ValueError("%s is not a valid --size." % FLAGS.size)
        infer_config.batch_size = 1
        infer_config.num_steps = 1

        word_to_id = {}
        id_to_word = {}

        if FLAGS.infer:
            vocab = vocab_pb2.VocabProto()
            with open(FLAGS.save_path + "/vocab.pbtxt", "rb") as f:
                text_format.Merge(f.read(), vocab)
                for i in vocab.item:
                    word_to_id[i.word] = i.id
                    id_to_word[i.id] = i.word

            vocab_size = len(word_to_id)

            graph_def = tf.GraphDef()
            with open(FLAGS.save_path + "/graph.pb", "rb") as f:
                graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")

            with tf.Session() as sess:
                sess.run(tf.initialize_all_variables())
                while True:
                    print "Sequence:"
                    seq_words = raw_input().split()
                    seq_ids = []
                    for w in seq_words:
                        if w in word_to_id:
                            seq_ids.append(word_to_id[w])
                        else:
                            print "'%s' was not seen in the training data." % w
                            seq_ids.append(word_to_id["<unk>"])
                    padded_input = np.pad(np.array([seq_ids]), ((0, 0), (0, infer_config.num_steps - len(seq_words))), 'constant', constant_values=0)
                    print predict(sess, padded_input, id_to_word, len(seq_words))


        else:
            input_data, word_to_id = reader.raw_data(FLAGS.training_data_path, train_config.min_frequency)
            id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
            epoch_size_scalar = ((len(input_data) // train_config.batch_size) - 1) // train_config.num_steps

            initialiser = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
            vocab_size = len(word_to_id)
            with tf.name_scope("training"):
                with tf.variable_scope("lstm", reuse=None, initializer=initialiser):
                    training_model = LSTM(train_config, vocab_size, epoch_size_scalar, is_training=True)
            with tf.name_scope("inference"):
                with tf.variable_scope("lstm", reuse=True, initializer=initialiser):
                    inference_model = LSTM(infer_config, vocab_size, epoch_size_scalar, is_training=False)

            with tf.Session() as sess:
                saver = tf.train.Saver()
                tf.initialize_all_variables().run()

                for i in xrange(train_config.max_max_epoch):
                    lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.0)
                    training_model.assign_lr(sess, train_config.lr * lr_decay)

                    train_perplexity = run_epoch(sess, training_model, input_data)
                    print "Epoch: %d, Train perplexity: %.3f" % (i + 1, train_perplexity)

                print "Saving model to %s" % FLAGS.save_path

                # Save checkpoint.
                saver.save(sess, os.path.join(FLAGS.save_path, "graph.ckpt"))

                # Save model for use in C++.
                vocab = vocab_pb2.VocabProto()
                vocab.min_frequency = train_config.min_frequency
                for i in id_to_word:
                    item = vocab.item.add()
                    item.id = i
                    item.word = id_to_word[i]
                with open(FLAGS.save_path + "/vocab.pbtxt", "wb") as f:
                    f.write(text_format.MessageToString(vocab))

                # Note: graph_util.convert_variables_to_constants() appends ':0' onto the variable names, which
                # is why it isn't included in 'inference/lstm/predictions'.
                graph_def = graph_util.convert_variables_to_constants(
                    sess=sess, input_graph_def=sess.graph.as_graph_def(), output_node_names=["inference/lstm/predictions"])

                tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pb", as_text=False)
                tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pbtxt")

if __name__ == "__main__":
    tf.app.run()
