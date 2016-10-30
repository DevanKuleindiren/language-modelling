import io
import lm.model


class Benchmark:

    def __init__(self, model):
        assert isinstance(model, lm.model.LM)
        self._model = model

    # TODO: Add cross-entropy, guessing-entropy and keys-saved.
    def accuracy(self, input_file):
        """Calculates accuracy metrics for the LM over the input_file.

        Args:
            input_file: The file containing the evaluation input.

        Returns:
            A dictionary of evaluation metrics, containing the perplexity of the language model.
        """
        word_seq = []
        perplexity = 1.0
        num_words = 0

        with open(input_file) as f:
            for line in f:
                words = line.split()
                for word in words:
                    num_words += 1
                    perplexity *= (1 / self._model.prob(word, word_seq))

                    if word == ".":
                        word_seq = []
                    else:
                        word_seq.append(word)

        return {"perplexity": perplexity ** (1 / num_words)}

    def timing(self):
        return 0.0
