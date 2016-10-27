import lm.ngram.model as ngram


class AddOne(ngram.NGramLM):

    def predict(self, word_seq):
        prediction = ""
        max_p = 0
        trimmed_word_seq = word_seq[-self._n+1:]

        for word in self._trie.vocab():
            numerator = self._trie.count(trimmed_word_seq + [word]) + 1
            denominator = self._trie.sum_following(trimmed_word_seq) + len(self._trie.vocab())

            p = numerator / float(denominator)

            print("P(%s) = %f" % (word, p))

            if p > max_p:
                max_p = p
                prediction = word

        return prediction
