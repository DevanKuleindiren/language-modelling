import lm.model


class Benchmark:

    def __init__(self, model, input_source):
        assert isinstance(model, lm.model.LM)
        self._model = model

    def perplexity(self):
        return 0.0

    def cross_entropy(self):
        return 0.0

    def guessing_entropy(self):
        return 0.0

    def keys_saved(self):
        return 0.0

    def timing(self):
        return 0.0
