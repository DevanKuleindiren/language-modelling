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

        print(word_seq)
        print(self._trie.words_following(word_seq))

        for word in self._trie.words_following(word_seq):
            if self._trie.count(word_seq + [word]) > max_count:
                prediction = word

        return prediction

    def parse_file(self, file_name):
        window = deque([])
        with open(file_name, "r") as f:
            count = 0
            for line in f:
                count += 1
                if count % 10000 == 0:
                    print("Parsed %d lines." % count)

                # Split the line into words.
                words = line.split(" ")
                if words:
                    words[-1] = words[-1].rstrip("\n")

                for word in words:
                    # Update the sliding n-gram window.
                    window.append(word)
                    if len(window) > self._n:
                        window.popleft()

                    # Insert all n-grams in the window.
                    n_gram = []
                    for word in reversed(window):
                        n_gram.insert(0, word)
                        self._trie.insert(n_gram)

                window = deque([])
