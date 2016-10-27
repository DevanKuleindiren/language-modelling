import lm.ngram.model as ngram


class KneserNey(ngram.NGramLM):

    def __init__(self, n, discount):
        super().__init__(n)
        self._discount = discount

    def predict(self, word_seq):
        prediction = ""
        max_p = 0
        trimmed_word_seq = word_seq[-self._n+1:]

        for word in self._trie.words_following(trimmed_word_seq):
            max_numerator = max(self._trie.count(trimmed_word_seq + [word]) - self._discount, 0)
            sum_following = float(self._trie.sum_following(trimmed_word_seq))
            num_following = self._trie.count_following(trimmed_word_seq)
            p = (max_numerator +
                 self._discount * num_following * self._prob_kn(word, word_seq[-self._n+2:])) / sum_following

            if p > max_p:
                max_p = p
                prediction = word

        return prediction

    def _prob_kn(self, word, word_seq):
        l = len(word_seq)
        assert l > 0
        assert l < self._n

        num_preceding = self._trie.count_preceding(word_seq)

        if l == 1:
            num_bigrams = float(self._trie.total_seqs_of_len(2))
            return num_preceding / num_bigrams

        num_preceding_and_following = float(self._trie.count_preceding_and_following(word_seq))
        num_following = self._trie.count_following(word_seq)

        max_numerator = max(num_preceding - self._discount, 0)

        return (max_numerator +
                self._discount * num_following * self._prob_kn(word, word_seq[1:])) / num_preceding_and_following
