import heapq
import reader
import numpy as np
import tensorflow as tf
import time

tf.flags.DEFINE_bool("infer", False, "Run inference on a previously saved model.")
tf.flags.DEFINE_string("training_data_path", None,"The path to the training data.")
tf.flags.DEFINE_string("save_path", None, "The path to save the model.")

FLAGS = tf.flags.FLAGS

class Config:

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
    - num_layers - the number of LSTM layers
    - num_steps - the number of unrolled steps of LSTM
    """
    batch_size = 10
    hidden_size = 10
    init_scale = 0.1
    keep_prob = 0.5
    lr = 1.0
    lr_decay = 0.8
    max_epoch = 1
    max_grad_norm = 5
    max_max_epoch = 1
    min_frequency = 1
    num_layers = 2
    num_steps = 10

class LSTM:

    def __init__(self, config, vocab_size, epoch_size, is_training):

        self._config = config
        self._epoch_size = epoch_size
        self._vocab_size = vocab_size
        self._input_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])
        self._target_data = tf.placeholder(tf.int32, [config.batch_size, config.num_steps])

        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
        if config.keep_prob < 1:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, dtype=tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("LSTM"):
          for time_step in range(config.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [config.hidden_size, vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        self._logits = logits
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(self._target_data, [-1])],
            [tf.ones([config.batch_size * config.num_steps], dtype=tf.float32)])
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

def predict(sess, model, inputs, id_to_word, seq_len):
    state = sess.run(model.initial_state)
    fetches = {
        "final_state": model.final_state,
        "logits": model.logits,
    }

    feed_dict = {}
    feed_dict[model.input_data] = inputs
    feed_dict[model.target_data] = np.zeros((model.config.batch_size, model.config.num_steps), dtype=np.int32)

    for i, (c, h) in enumerate(model.initial_state):
        feed_dict[c] = state[i].c
        feed_dict[h] = state[i].h
    vals = sess.run(fetches, feed_dict)
    state = vals["final_state"]
    logits = vals["logits"]

    return sorted(heapq.nlargest(10, [(id_to_word[i], p) for (i, p) in zip(xrange(model.vocab_size), logits[seq_len])], key=lambda x: x[1]), key=lambda x: -x[1])


def main(_):
    if not FLAGS.training_data_path:
        raise ValueError("Must set --training_data_path.")

    with tf.Graph().as_default():
        config = Config()
        input_data, word_to_id = reader.raw_data(FLAGS.training_data_path, config.min_frequency)
        vocab_size = len(word_to_id)
        id_to_word = dict(zip(word_to_id.values(), word_to_id.keys()))
        epoch_size_scalar = ((len(input_data) // config.batch_size) - 1) // config.num_steps

        initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        with tf.variable_scope("Model", reuse=None, initializer=initialiser):
            training_model = LSTM(config, vocab_size, epoch_size_scalar, is_training=True)
        with tf.variable_scope("Model", reuse=True, initializer=initialiser):
            inference_model = LSTM(config, vocab_size, epoch_size_scalar, is_training=False)

        with tf.Session() as sess:
            saver = tf.train.Saver()
            tf.initialize_all_variables().run()

            if FLAGS.infer:
                ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
                print (ckpt)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                else:
                    print ("No checkpoint file found")

                print "Sequence:"
                seq_words = raw_input().split()
                seq_ids = []
                for w in seq_words:
                    if w in word_to_id:
                        seq_ids.append(word_to_id[w])
                    else:
                        print "'%s' was not seen in the training data." % w
                        seq_ids.append(word_to_id["<unk>"])
                padded_input = np.pad(np.array([seq_ids]), ((0, config.batch_size - 1), (0, config.num_steps - len(seq_words))), 'constant', constant_values=0)
                print padded_input
                print predict(sess, inference_model, padded_input, id_to_word, len(seq_words))

            else:
                for i in xrange(config.max_max_epoch):
                    lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                    training_model.assign_lr(sess, config.lr * lr_decay)

                    train_perplexity = run_epoch(sess, training_model, input_data)
                    print "Epoch: %d, Train perplexity: %.3f" % (i + 1, train_perplexity)

                    if FLAGS.save_path:
                        print "Saving model to %s" % FLAGS.save_path
                        saver.save(sess, FLAGS.save_path+'/model.ckpt', global_step=i+1)


if __name__ == "__main__":
    tf.app.run()
