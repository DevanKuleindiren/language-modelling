import reader
import numpy as np
import tensorflow as tf
import time


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
    hidden_size = 50
    init_scale = 0.1
    keep_prob = 0.5
    lr = 1.0
    lr_decay = 0.8
    max_epoch = 5
    max_grad_norm = 5
    max_max_epoch = 20
    min_frequency = 1
    num_layers = 2
    num_steps = 20
    vocab_size = 10000

class RNN:

    def __init__(self, config, input_data, target_data, epoch_size):

        self._config = config
        self._epoch_size = epoch_size

        rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(config.hidden_size, forget_bias=0.0, state_is_tuple=True)
        if config.keep_prob < 1:
            rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell, output_keep_prob=config.keep_prob)
        cell = tf.nn.rnn_cell.MultiRNNCell([rnn_cell] * config.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(config.batch_size, dtype=tf.float32)

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [config.vocab_size, config.hidden_size], dtype=tf.float32)
            inputs = tf.nn.embedding_lookup(embedding, input_data)

        if config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, keep_prob=config.keep_prob)

        outputs = []
        state = self._initial_state
        with tf.variable_scope("RNN"):
          for time_step in range(config.num_steps):
            if time_step > 0: tf.get_variable_scope().reuse_variables()
            (cell_output, state) = cell(inputs[:, time_step, :], state)
            outputs.append(cell_output)

        output = tf.reshape(tf.concat(1, outputs), [-1, config.hidden_size])
        softmax_w = tf.get_variable(
            "softmax_w", [config.hidden_size, config.vocab_size], dtype=tf.float32)
        softmax_b = tf.get_variable("softmax_b", [config.vocab_size], dtype=tf.float32)
        logits = tf.matmul(output, softmax_w) + softmax_b
        loss = tf.nn.seq2seq.sequence_loss_by_example(
            [logits],
            [tf.reshape(target_data, [-1])],
            [tf.ones([config.batch_size * config.num_steps], dtype=tf.float32)])
        self._cost = cost = tf.reduce_sum(loss) / config.batch_size
        self._final_state = state

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
    def initial_state(self):
        return self._initial_state

    @property
    def config(self):
        return self._config

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def train_op(self):
        return self._train_op

    @property
    def epoch_size(self):
        return self._epoch_size



def run_epoch(sess, model):
    start_time = time.time()
    costs = 0.0
    iters = 0
    state = sess.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
        "train_op": model.train_op,
    }

    for step in range(model.epoch_size):
        feed_dict = {}

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = sess.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost
        iters += model.config.num_steps

        if step % (model.epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                (step * 1.0 / model.epoch_size, np.exp(costs / iters),
                 iters * model.config.batch_size / (time.time() - start_time)))

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.training_data_path:
        raise ValueError("Must set --training_data_path.")

    with tf.Graph().as_default():
        config = Config()
        inputs, targets, epoch_size_scalar = reader.batch_producer(FLAGS.training_data_path, config.batch_size, config.num_steps, config.min_frequency)

        initialiser = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

        model = RNN(config, inputs, targets, epoch_size_scalar)

        sv = tf.train.Supervisor(logdir=FLAGS.save_path)
        with sv.managed_session() as sess:
            for i in xrange(config.max_max_epoch):
                lr_decay = config.lr_decay ** max(i + 1 - config.max_epoch, 0.0)
                model.assign_lr(sess, config.lr * lr_decay)

                train_perplexity = run_epoch(sess, model)
                print "Epoch: %d, Train perplexity: %.3f" % (i + 1, train_perplexity)

            if FLAGS.save_path:
                print "Saving model to %s" % FLAGS.save_path
                sv.saver.save(sess, FLAGS.save_path, global_step=sv.global_step)


if __name__ == "__main__":
    tf.app.run()
