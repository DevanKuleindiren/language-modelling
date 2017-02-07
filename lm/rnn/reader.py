import collections
import numpy as np
import tensorflow as tf


def _get_data(file_name):
    with tf.gfile.GFile(file_name, "r") as f:
        return f.read().decode("utf-8").replace("\n", " <s> ").split()

def _word_to_id(file_name, min_frequency):
    data = _get_data(file_name)
    word_to_id = {}
    word_to_id["<unk>"] = 0
    word_to_id["<s>"] = 1
    id_ = 2

    word_counts = {}
    for word in data:
        if word not in word_counts:
            word_counts[word] = 0
        word_counts[word] += 1

        if word_counts[word] == min_frequency and word not in word_to_id:
            word_to_id[word] = id_
            id_ += 1

    return word_to_id

def raw_data(file_name, min_frequency, word_to_id=None):
    if not word_to_id:
        word_to_id = _word_to_id(file_name, min_frequency)
    raw_words = _get_data(file_name)
    raw_ids = [word_to_id[word] for word in raw_words if word in word_to_id]

    return raw_ids, word_to_id

def batch_producer(raw_data, batch_size, num_steps, name=None):
    epoch_size_scalar = ((len(raw_data) // batch_size) - 1) // num_steps

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    epoch_size = (batch_len - 1) // num_steps

    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in xrange(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in xrange(epoch_size):
        x = data[:, i * num_steps:(i + 1) * num_steps]
        y = data[:, i * num_steps + 1:(i + 1) * num_steps + 1]
        yield (x, y)
