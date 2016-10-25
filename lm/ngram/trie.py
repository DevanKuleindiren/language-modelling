from collections import deque

class Trie:

    def __init__(self):
        self._root = Node()
        self._total_counts = {}

    def _increment_total_count(self, n):
        if not n in self._total_counts:
            self._total_counts[n] = 0
        self._total_counts[n] += 1

    def insert(self, ngram):
        if ngram:
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
        to_visit = deque([(0, self._root)])
        result = ""

        result += "digraph TRIE {\n"
        count = 0
        while to_visit:
            c, node = to_visit.popleft()
            result += "    %d [label=\"%d\"];\n" % (c, node.get_count())
            for child in node.get_children():
                count += 1
                result += "    %d -> %d [label=\"%s\"];\n" % (c, count, child)
                to_visit.append((count, node.get_child(child)))
        result += "}\n"
        return result

    def print_ngrams(self):
        ngram_dict = {}
        self.dfs(self._root, [], ngram_dict)

        for ngram in sorted(ngram_dict):
            print("%s: %d" % (ngram, ngram_dict[ngram]))

    def dfs(self, node, path, ngram_dict):
        ngram_dict[' '.join(path)] = node.get_count()

        for child in node.get_children():
            self.dfs(node.get_children()[child], path + [child], ngram_dict)


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