import abc

class LM:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, word_seq):
        pass

    @abc.abstractmethod
    def predict_top_k(self, word_seq, n):
        pass

    @abc.abstractmethod
    def prob(self, word, word_seq):
        pass