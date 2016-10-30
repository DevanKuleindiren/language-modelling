import collections
import lm.ngram.trie as trie
import unittest


class TrieTest(unittest.TestCase):

    def setUp(self):
        self._under_test = trie.Trie()
        self._n = 3
        words = ["the", "cat", "sat", "on", "the", "mat", "sat", "on", "the", "floor", "."]
        window = collections.deque([])
        for word in words:
            window.append(word)
            if len(window) > self._n:
                window.popleft()
            n_gram = []
            for word in reversed(window):
                n_gram.insert(0, word)
                self._under_test.insert(n_gram)

    def test_insert_rejects_non_list(self):
        with self.assertRaises(AssertionError):
            under_test = trie.Trie()
            under_test.insert(10)

    def test_insert(self):
        under_test = trie.Trie()
        under_test.insert(["the"])
        self.assertEqual(under_test._root.get_child("the").get_count(), 1)
        self.assertEqual(under_test._total_sums[1], 1)
        under_test.insert(["the", "cat"])
        self.assertEqual(under_test._root.get_child("the").get_child("cat").get_count(), 1)

    def test_count(self):
        self.assertEqual(self._under_test.count([]), 0)
        self.assertEqual(self._under_test.count(["blah"]), 0)
        self.assertEqual(self._under_test.count(["the"]), 3)
        self.assertEqual(self._under_test.count(["cat"]), 1)
        self.assertEqual(self._under_test.count(["sat"]), 2)
        self.assertEqual(self._under_test.count(["the", "cat"]), 1)
        self.assertEqual(self._under_test.count(["cat", "sat"]), 1)
        self.assertEqual(self._under_test.count(["sat", "on"]), 2)
        self.assertEqual(self._under_test.count(["the", "cat", "sat"]), 1)
        self.assertEqual(self._under_test.count(["sat", "on", "the"]), 2)
        self.assertEqual(self._under_test.count(["on", "the", "floor"]), 1)

    def test_count_following(self):
        self.assertEqual(self._under_test.count_following([]), 0)
        self.assertEqual(self._under_test.count_following(["blah"]), 0)
        self.assertEqual(self._under_test.count_following(["."]), 0)
        self.assertEqual(self._under_test.count_following(["the"]), 3)
        self.assertEqual(self._under_test.count_following(["cat"]), 1)
        self.assertEqual(self._under_test.count_following(["mat", "sat"]), 1)
        self.assertEqual(self._under_test.count_following(["on", "the"]), 2)
        self.assertEqual(self._under_test.count_following(["on", "the", "floor"]), 0)

    def test_count_preceding(self):
        self.assertEqual(self._under_test.count_preceding([]), 0)
        self.assertEqual(self._under_test.count_preceding(["blah"]), 0)
        self.assertEqual(self._under_test.count_preceding(["cat"]), 1)
        self.assertEqual(self._under_test.count_preceding(["the"]), 1)
        self.assertEqual(self._under_test.count_preceding(["sat"]), 2)

    def test_count_preceding_and_following(self):
        self.assertEqual(self._under_test.count_preceding_and_following([]), 0)
        self.assertEqual(self._under_test.count_preceding_and_following(["blah"]), 0)
        self.assertEqual(self._under_test.count_preceding_and_following(["cat"]), 1)
        self.assertEqual(self._under_test.count_preceding_and_following(["the"]), 2)

    def test_sum_following(self):
        self.assertEqual(self._under_test.sum_following([]), 0)
        self.assertEqual(self._under_test.sum_following(["blah"]), 0)
        self.assertEqual(self._under_test.sum_following(["cat"]), 1)
        self.assertEqual(self._under_test.sum_following(["on", "the"]), 2)
        self.assertEqual(self._under_test.sum_following(["the"]), 3)

    def test_words_following(self):
        self.assertEqual(collections.Counter(self._under_test.words_following(["the"])), collections.Counter(["cat",
                                                                                                           "mat",
                                                                                                            "floor"]))

    def test_vocab(self):
        self.assertEqual(collections.Counter(self._under_test.vocab()), collections.Counter(["the", "cat", "sat",
                                                                                            "on", "mat", "floor", "."]))

    def test_vocab_size(self):
        self.assertEqual(self._under_test.vocab_size(), len(self._under_test.vocab()))

    def test_total_seqs_of_len(self):
        self.assertEqual(self._under_test.total_seqs_of_len(-1), 0)
        self.assertEqual(self._under_test.total_seqs_of_len(4), 0)
        self.assertEqual(self._under_test.total_seqs_of_len(0), 0)
        self.assertEqual(self._under_test.total_seqs_of_len(1), 11)
        self.assertEqual(self._under_test.total_seqs_of_len(2), 10)
        self.assertEqual(self._under_test.total_seqs_of_len(3), 9)


class NodeTest(unittest.TestCase):

    def test_inc(self):
        under_test = trie.Node()
        self.assertEqual(under_test.get_count(), 0)
        under_test.inc()
        self.assertEqual(under_test.get_count(), 1)
        under_test.inc()
        under_test.inc()
        under_test.inc()
        self.assertEqual(under_test.get_count(), 4)

    def test_add_child(self):
        under_test = trie.Node()
        with self.assertRaises(KeyError):
            under_test.get_child("blah")

        under_test.add_child("the")
        self.assertIsInstance(under_test.get_child("the"), trie.Node)
        self.assertIsInstance(under_test.get_children(), dict)

if __name__ == '__main__':
    unittest.main()
