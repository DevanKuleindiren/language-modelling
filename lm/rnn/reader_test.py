"""Tests for tensorflow.models.ptb_lstm.ptb_reader."""

import os.path

import tensorflow as tf

from tensorflow.Source.lm.rnn import reader

class ReaderTest(tf.test.TestCase):

    def testRawData(self):
        tmpdir = tf.test.get_temp_dir()

        file_name = os.path.join(tmpdir, "test_file")
        with tf.gfile.GFile(file_name, "w") as f:
            f.write("\n".join(
                [" the cat sat on the mat . ",
                " the cat ate the mouse . ",
                " the dog sat on the cat . "]))

        expected_word_to_id = {
            "<unk>": 0,
            "<s>": 1,
            "the": 2,
            "cat": 3,
            "sat": 4,
            "on": 5,
            "mat": 6,
            ".": 7,
            "ate": 8,
            "mouse": 9,
            "dog": 10,
        }
        expected_raw_ids = [2, 3, 4, 5, 2, 6, 7, 1, 2, 3, 8, 2, 9, 7, 1, 2, 10, 4, 5, 2, 3, 7]

        actual_raw_ids, actual_word_to_id = reader.raw_data(file_name, 1)

        self.assertEqual(actual_word_to_id, expected_word_to_id)
        self.assertEqual(actual_raw_ids, expected_raw_ids)


if __name__ == "__main__":
  tf.test.main()
