import collections
import tensorflow as tf


def _get_data(file_name):
    with tf.gfile.GFile(file_name, "r") as f:
        return f.read().decode("utf-8").replace("\n", " <s> ").split()

def _word_to_id(file_name, min_frequency):
    data = _get_data(file_name)
    word_to_id = {}
    word_to_id["<unk>"] = 0

    counter = collections.Counter(data)
    words = [w for w, c in counter.iteritems() if c >= min_frequency]
    return dict(zip(words, range(len(words))))


def batch_producer(file_name, batch_size, num_steps, min_frequency, name=None):
    word_to_id = _word_to_id(file_name, min_frequency)
    raw_words = _get_data(file_name)
    raw_ids = [word_to_id[word] for word in raw_words if word in word_to_id]

    epoch_size_scalar = ((len(raw_ids) // batch_size) - 1) // num_steps

    with tf.name_scope(name, "batch_producer", [raw_ids, batch_size, num_steps]):
        raw_data = tf.convert_to_tensor(raw_ids, name="training_data", dtype=tf.int32)

        data_len = tf.size(raw_data)
        batch_len = data_len // batch_size
        epoch_size = (batch_len - 1) // num_steps
        data = tf.reshape(raw_data[0 : batch_size * batch_len], [batch_size, batch_len])

        assertion = tf.assert_positive(
            epoch_size,
            message="epoch_size == 0, decrease batch_size or num_steps")

        with tf.control_dependencies([assertion]):
            epoch_size = tf.identity(epoch_size, name="epoch_size")

        i = tf.train.range_input_producer(epoch_size, shuffle=False).dequeue()
        x = tf.slice(data, [0, i * num_steps], [batch_size, num_steps])
        y = tf.slice(data, [0, i * num_steps + 1], [batch_size, num_steps])

        return x, y, epoch_size_scalar
