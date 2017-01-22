import os
import tensorflow as tf

from google.protobuf import text_format
from tensorflow.python.framework import graph_util
from tensorflow.Source.lm import vocab_pb2

tf.flags.DEFINE_string("save_path", None, "The path to save the model.")

FLAGS = tf.flags.FLAGS

def main(_):
    if not FLAGS.save_path:
        print "You must specify --save_path."

    with tf.name_scope("inference/lstm"):
        input_data = tf.placeholder(tf.int32, [1, 1], name="inputs")
        input_data_f = tf.to_float(input_data)
        initial_state_c_0 = tf.zeros([1, 3], name="zeros")
        initial_state_h_0 = tf.zeros([1, 3], name="zeros_1")
        initial_state_c_1 = tf.zeros([1, 3], name="zeros_2")
        initial_state_h_1 = tf.zeros([1, 3], name="zeros_3")
        w = tf.constant([[0.2, 0.5, 0.3]])
        final_state_c_0 = tf.add(initial_state_c_0 + 0.1, input_data_f, name="RNN/MultiRNNCell/Cell0/BasicLSTMCell/add_2")
        final_state_h_0 = tf.add(initial_state_h_0 + 0.2, input_data_f, name="RNN/MultiRNNCell/Cell0/BasicLSTMCell/mul_2")
        final_state_c_1 = tf.add(initial_state_c_1 + 0.3, input_data_f, name="RNN/MultiRNNCell/Cell1/BasicLSTMCell/add_2")
        final_state_h_1 = tf.add(initial_state_h_1 + 0.4, input_data_f, name="RNN/MultiRNNCell/Cell1/BasicLSTMCell/mul_2")
        y = final_state_c_0 + final_state_h_0 + final_state_c_1 + final_state_h_1 - (3 * input_data_f)
        predictions = tf.mul(tf.matmul(input_data_f, w), y, name="predictions")

    with tf.Session() as sess:
        tf.initialize_all_variables().run()

        # Save model for use in C++.
        vocab = vocab_pb2.VocabProto()
        vocab.min_frequency = 1
        word_to_id = {
            "<unk>": 0,
            "<s>": 1,
            "the": 2,
        }
        for w in word_to_id:
            item = vocab.item.add()
            item.id = word_to_id[w]
            item.word = w
        with open(os.path.join(FLAGS.save_path, "vocab.pbtxt"), "wb") as f:
            f.write(text_format.MessageToString(vocab))

        # Note: graph_util.convert_variables_to_constants() appends ':0' onto the variable names, which
        # is why it isn't included in 'inference/lstm/predictions'.
        graph_def = graph_util.convert_variables_to_constants(
            sess=sess, input_graph_def=sess.graph.as_graph_def(), output_node_names=["inference/lstm/predictions"])

        tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pb", as_text=False)
        tf.train.write_graph(graph_def, FLAGS.save_path, "graph.pbtxt")

if __name__ == "__main__":
    tf.app.run()
