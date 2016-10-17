import abc

class LM:
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def predict(self, word_seq):
        pass
