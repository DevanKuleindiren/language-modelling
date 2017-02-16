import numpy as np
import os
import reader
import tensorflow as tf
import time

from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.Source.lm import vocab_pb2
from tensorflow.Source.lm.rnn import rnn_pb2

tf.flags.DEFINE_string("type", None, "The type of RNN you want to train (one of: rnn, gru, lstm).")
tf.flags.DEFINE_string("training_data_path", None, "The file path of the training data.")
tf.flags.DEFINE_string("test_data_path", None, "The file path of the test data.")
tf.flags.DEFINE_string("save_path", None, "The directory path to save the model in.")
tf.flags.DEFINE_string("size", None, "The size of the model (one of: small, large).")

FLAGS = tf.flags.FLAGS


class SmallConfig:
    """
    The hyperparameters used in the model:
    - batch_size - The batch size.
    - hidden_size - The number of RNN units.
    - init_scale - The initial scale of the weights.
    - keep_prob - The probability of keeping weights in the dropout layer.
    - lr - The initial value of the learning rate.
    - lr_decay - The decay of the learning rate for each epoch after "max_epoch".
    - max_grad_norm - The maximum permissible norm of the gradient.
    - max_epoch - The number of epochs trained with the initial learning rate.
    - max_max_epoch - The total number of epochs for training.
    - min_frequency - The minimum number of times a word needs to be seen to be considered part of the vocabulary.
    - num_layers - The number of RNN layers.
    - num_steps - The number of unrolled steps of the RNN for each chunk of the training data.
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


class CellType:
    VANILLA = 0
    GRU = 1
    LSTM = 2


class RNN:

    def __init__(self, config, cell_type, vocab_size, epoch_size, is_training):

        self._config = config
        self._cell_type = cell_type
        self._epoch_size = epoch_size
        self._vocab_size = vocab_size

        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name="inputs")
        self._target_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps], name="targets")

        # A 'cell' in TensorFlow actually refers to an array of the RNNs cells described in literature, so this is an
        # array of config.hidden_size RNN cells.
        if cell_type == CellType.VANILLA:
            rnn_cell = tf.nn.rnn_cell.BasicRNNCell(config.hidden_size)
        elif cell_type == CellType.GRU:
            rnn_cell = tf.nn.rnn_cell.GRUCell(config.hidden_size)
        else:
            rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)

        # This adds dropout to the RNN cells by applying the probability of keeping a weight.
        if is_training and config.keep_prob < 1:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)

        # This adds layers to the RNN to give config.num_layers layers overall.
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)

        # Initialise the RNN cell weights.
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
        with tf.variable_scope("RNN"):
            for time_step in range(config.num_steps):
                # We want to use the same set of weights in each time step. This is due to the recurrent structure of
                # RNNs.
                if time_step > 0: tf.get_variable_scope().reuse_variables()

                # Each RNN cell has a __call__ function which takes a batch of words at a particular time step
                # (represented as embedding vectors) and returns the tuple (h, s) where h is the new RNN activation
                # and s is the new cell state. In the case of the vanilla RNN and GRU architectures,this state is just
                # the activations from the previous time step. In the case of the LSTM architecture, this state is a
                # tuple (c, h), which corresponds to c and h in Figure 1 of http://arxiv.org/abs/1409.2329.
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

                # Each RNN cell activation is stored in a list of outputs.
                outputs.append(cell_output)

        # Concatenate the outputs from each time step along dimension 1 (which should give a matrix of shape
        # ((config.batch_size * config.num_steps) x config.hidden_size).
        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])

        softmax_w = tf.get_variable(
            "softmax_w", [config.hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        # Multiply the RNN activations and add bias to give logits of shape
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

    @property
    def inputs(self):
        return self._input_data

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def final_state(self):
        return self._final_state

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def run_epoch(self, sess, input_data, is_training=True):
        start_time = time.time()
        costs = 0.0
        iters = 0
        state = sess.run(self._initial_state)

        fetches = {
            "cost": self._cost,
            "final_state": self._final_state,
        }
        if is_training:
            fetches["train_op"] = self._train_op

        for step, (x, t) in enumerate(reader.batch_producer(input_data, self._config.batch_size, self._config.num_steps)):
            feed_dict = {}
            feed_dict[self._input_data] = x
            feed_dict[self._target_data] = t

            if self._cell_type == CellType.VANILLA or self._cell_type == CellType.GRU:
                for i, h in enumerate(self._initial_state):
                    feed_dict[h] = state[i]
            else:
                # For the LSTM model, the states are tuples (c, h).
                for i, (c, h) in enumerate(self._initial_state):
                    feed_dict[c] = state[i].c
                    feed_dict[h] = state[i].h

            vals = sess.run(fetches, feed_dict)
            cost = vals["cost"]
            state = vals["final_state"]

            costs += cost
            iters += self._config.num_steps

            if is_training and step % (self._epoch_size // 10) == 10:
                print("%.3f perplexity: %.3f speed: %.0f wps" %
                    (step * 1.0 / self._epoch_size, np.exp(costs / iters),
                     iters * self._config.batch_size / (time.time() - start_time)))

        return np.exp(costs / iters)

def main(_):
    if not FLAGS.type:
        raise ValueError("Must set --type.")
    if not FLAGS.training_data_path:
        raise ValueError("Must set --training_data_path.")
    if not FLAGS.save_path:
        raise ValueError("Must set --save_path.")
    if not FLAGS.size:
        raise ValueError("Must set --size.")

    with tf.Graph().as_default():
        if FLAGS.type == "rnn":
            cell_type = CellType.VANILLA
        elif FLAGS.type == "gru":
            cell_type = CellType.GRU
        elif FLAGS.type == "lstm":
            cell_type = CellType.LSTM
        else:
            raise ValueError("%s is not a valid --type." % FLAGS.type)

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

        train_data, word_to_id = reader.raw_data(FLAGS.training_data_path, train_config.min_frequency)
        id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
        epoch_size_scalar = ((len(train_data) // train_config.batch_size) - 1) // train_config.num_steps

        initialiser = tf.random_uniform_initializer(-train_config.init_scale, train_config.init_scale)
        vocab_size = len(word_to_id)
        with tf.name_scope("training"):
            with tf.variable_scope("rnn", reuse=None, initializer=initialiser):
                training_model = RNN(train_config, cell_type, vocab_size, epoch_size_scalar, is_training=True)
        with tf.name_scope("inference"):
            with tf.variable_scope("rnn", reuse=True, initializer=initialiser):
                inference_model = RNN(infer_config, cell_type, vocab_size, epoch_size_scalar, is_training=False)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()
            start_time = time.time()

            for i in xrange(train_config.max_max_epoch):
                lr_decay = train_config.lr_decay ** max(i + 1 - train_config.max_epoch, 0.0)
                training_model.assign_lr(sess, train_config.lr * lr_decay)

                train_perplexity = training_model.run_epoch(sess, train_data)
                print "Epoch: %d, Train perplexity: %.3f" % (i + 1, train_perplexity)

            print "Trained in %d seconds." % (time.time() - start_time)

            if FLAGS.test_data_path:
                test_data, _ = reader.raw_data(FLAGS.test_data_path, train_config.min_frequency, word_to_id)
                test_perplexity = training_model.run_epoch(sess, test_data, is_training=False)
                print "Test perplexity: %.3f" % test_perplexity

            print "Saving model to %s" % FLAGS.save_path

            # Save checkpoint.
            saver.save(sess, os.path.join(FLAGS.save_path, "graph.ckpt"))

            # Save model for use in C++.
            # --------------------------

            # Save meta information about the RNN.
            rnn_proto = rnn_pb2.RNNProto()
            rnn_proto.type = cell_type
            rnn_proto.input_tensor_name = inference_model.inputs.name
            rnn_proto.logits_tensor_name = inference_model.logits.name
            rnn_proto.predictions_tensor_name = inference_model.predictions.name
            if cell_type == CellType.VANILLA or cell_type == CellType.GRU:
                for i, (h_0, h_1) in enumerate(zip(inference_model.initial_state, inference_model.final_state)):
                    h_name_pair = rnn_proto.h.add()
                    h_name_pair.initial = h_0.name
                    h_name_pair.final = h_1.name
            else:
                for i, ((c_0, h_0), (c_1, h_1)) in enumerate(zip(inference_model.initial_state, inference_model.final_state)):
                    h_name_pair = rnn_proto.h.add()
                    h_name_pair.initial = h_0.name
                    h_name_pair.final = h_1.name

                    c_name_pair = rnn_proto.c.add()
                    c_name_pair.initial = c_0.name
                    c_name_pair.final = c_1.name
            with open(os.path.join(FLAGS.save_path, "rnn.pbtxt"), "wb") as f:
                f.write(text_format.MessageToString(rnn_proto))

            # Save the vocabulary.
            vocab = vocab_pb2.VocabProto()
            vocab.min_frequency = train_config.min_frequency
            for i in id_to_word:
                item = vocab.item.add()
                item.id = i
                item.word = id_to_word[i]
            with open(os.path.join(FLAGS.save_path, "vocab.pbtxt"), "wb") as f:
                f.write(text_format.MessageToString(vocab))

            # Note: graph_util.convert_variables_to_constants() appends ':0' onto the variable names, which
            # is why it isn't included in 'inference/rnn/predictions'.
            graph_def = graph_util.convert_variables_to_constants(
                sess=sess, input_graph_def=sess.graph.as_graph_def(),
                output_node_names=[inference_model.predictions.name.split(':', 1)[0]])

            # Save the frozen graph (which can be loaded into C++).
            tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pb", as_text=False)
            tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pbtxt")

if __name__ == "__main__":
    tf.app.run()
