import lm.ngram.model as ngram


class AddOne(ngram.NGramLM):

    def prob(self, word, word_seq):
        trimmed_word_seq = word_seq[-self._n+1:]
        return self._trie.count(trimmed_word_seq + [word]) + 1 / float(self._trie.sum_following(trimmed_word_seq) +
                                                                       len(self._trie.vocab()))
