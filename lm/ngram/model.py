from collections import deque
import heapq
import lm.model as lm
import lm.ngram.trie as trie


class NGramLM(lm.LM):

    def __init__(self, n):
        self._n = n
        self._trie = trie.Trie()

    def predict(self, word_seq):
        top_prediction = self.predict_top_k(word_seq, 1)
        return top_prediction[0]

    def predict_top_k(self, word_seq, k):
        if k < 1:
            return []

        max_predictions = []
        for word in self._trie.vocab():
            p = self.prob(word, word_seq)
            if len(max_predictions) > k:
                min_max_p, _ = max_predictions[0]
                if p > min_max_p:
                    heapq.heapreplace(max_predictions, (p, word))
            else:
                heapq.heappush(max_predictions, (p, word))

        final_top_k = []
        while max_predictions:
            final_top_k.insert(0, heapq.heappop(max_predictions))

        return final_top_k

    def prob(self, word, word_seq):
        trimmed_word_seq = word_seq[-self._n+1:]
        return self._trie.count(trimmed_word_seq + [word]) / float(self._trie.count_following(trimmed_word_seq))

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
