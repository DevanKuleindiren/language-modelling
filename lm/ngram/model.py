from collections import deque

import lm.model as lm
import lm.ngram.trie as trie


class NGramLM(lm.LM):

    def __init__(self, n):
        self._n = n
        self._trie = trie.Trie()

    def predict(self, word_seq):
        prediction = ""
        max_count = 0

        end_word_node = self._trie.get_node(word_seq[(-self._n + 1):])
        if end_word_node:
            for child in end_word_node._children:
                if end_word_node._children[child]._count > max_count:
                    max_count = end_word_node._children[child]._count
                    prediction = child

        return prediction

    def parse_file(self, file_name):
        window = deque([])
        with open(file_name, "r") as f:
            count = 0
            for line in f:

                count += 1
                if count % 10000 == 0:
                    print("Parsed %d lines." % count)

                words = line.split(" ")
                if words:
                    words[-1] = words[-1].rstrip("\n")
                for word in words:
                    window.append(word)
                    if len(window) > self._n:
                        window.popleft()
                    self._trie.insert_all(list(window))
                window = deque([])
