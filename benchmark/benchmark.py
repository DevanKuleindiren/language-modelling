import lm.model
import math


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
        product = 1.0
        products = []
        num_words = 0
        num_lines = 0

        with open(input_file) as f:

            for line in f:
                words = line.split()
                for word in words:
                    fraction = 1 / self._model.prob(word, word_seq)
                    new_product = product * fraction
                    if math.isinf(new_product):
                        products.append(product)
                        product = fraction
                    else:
                        product = new_product

                    if word == ".":
                        word_seq = []
                    else:
                        word_seq.append(word)
                    num_words += 1

                num_lines += 1
                if num_lines % 10 == 0:
                    print("Evaluated %d lines." % num_lines)

        products.append(product)
        perplexity = 1.0
        for product in products:
            perplexity *= math.pow(product, 1 / num_words)

        return {"perplexity": perplexity}

    def timing(self):
        return 0.0
