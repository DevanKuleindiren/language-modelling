from collections import deque
import experimental.lm as lm

class NGramLM(lm.LM):

    def __init__(self, n):
        self._n = n
        self._trie = Trie()

    def predict(self, word_seq):
        prediction = ""
        max_count = 0

        end_word_node = self._trie.get_node(word_seq[(-self._n + 1):])
        if end_word_node:
            for child in end_word_node._children:
                if end_word_node._children[child]._count > max_count:
                    max_count = end_word_node._children[child]._count
                    prediction = child

        return prediction

    def parse_file(self, file_name):
        window = deque([])
        with open(file_name, "r") as f:
            count = 0
            for line in f:

                count += 1
                if count % 10000 == 0:
                    print("Parsed %d lines." % count)

                words = line.split(" ")
                if words:
                    words[-1] = words[-1].rstrip("\n")
                for word in words:
                    window.append(word)
                    if len(window) > self._n:
                        window.popleft()
                    self._trie.insert_all(list(window))
                window = deque([])

class Trie:

    def __init__(self):
        self._root = Node()
        self._total_counts = {}

    def _increment_total_count(self, n):
        if not n in self._total_counts:
            self._total_counts[n] = 0
        self._total_counts[n] += 1

    def insert_all(self, word_seq):
        n_gram = []

        for word in reversed(word_seq):
            n_gram.insert(0, word)
            self.insert_ngram(n_gram)

    def insert_ngram(self, ngram):
        if ngram != []:
            node = self._root
            last_word = ngram[-1]

            for word in ngram[:-1]:
                if not node.has_child(word):
                    node.add_child(word)
                node = node.get_child(word)

            if not node.has_child(last_word):
                node.add_child(last_word)
            node.get_child(last_word).inc()
        self._increment_total_count(len(ngram))

    def get_node(self, word_seq):
        node = self._root

        for word in word_seq:
            if not node.has_child(word):
                return None
            node = node.get_child(word)

        return node

    def __str__(self):
        to_visit = deque([("ROOT", self._root)])
        result = ""

        while to_visit:
            name, node = to_visit.popleft()
            for child in node._children:
                result += "%s -> %s\n" % (name, child)
                to_visit.append((child, node._children[child]))
            result += "---\n"
        result += "***\n"

        return result

    def print_ngrams(self):
        ngram_dict = {}
        self.dfs(self._root, [], ngram_dict)

        for ngram in sorted(ngram_dict):
            print("%s: %d" % (ngram, ngram_dict[ngram]))

    def dfs(self, node, path, ngram_dict):
        ngram_dict[' '.join(path)] = node._count

        for child in node._children:
            self.dfs(node._children[child], path + [child], ngram_dict)

class Node:
    def __init__(self):
        self._children = {}
        self._count = 0

    def has_child(self, word):
        return word in self._children

    def add_child(self, word):
        self._children[word] = Node()

    def get_child(self, word):
        return self._children[word]

    def get_children(self):
        return self._children

    def inc(self):
        self._count += 1

    def get_count(self):
        return self._count