from collections import deque


class Trie:

    def __init__(self):
        self._root = Node()
        self._total_sums = {}
        self._vocab_size = 0

    def count(self, seq):
        node = self._get_node(seq)
        if node:
            return node.get_count()
        return 0

    def count_following(self, seq):
        if seq:
            node = self._get_node(seq)
            if node:
                return len(node.get_children())
        return 0

    def count_preceding(self, seq):
        count = 0
        if seq:
            for start_word in self._root.get_children():
                if self.count([start_word] + seq) > 0:
                    count += 1
        return count

    def count_preceding_and_following(self, seq):
        count = 0
        if seq:
            for start_word in self._root.get_children():
                count += self.count_following([start_word] + seq)
        return count

    def sum_following(self, seq):
        count = 0
        if seq:
            node = self._get_node(seq)
            if node:
                for child in node.get_children():
                    count += node.get_child(child).get_count()
        return count

    def words_following(self, seq):
        if seq:
            node = self._get_node(seq)
            if node:
                return list(node.get_children().keys())
        return []

    def vocab(self):
        return list(self._root.get_children().keys())

    def vocab_size(self):
        return self._vocab_size

    def total_seqs_of_len(self, n):
        if n in self._total_sums:
            return self._total_sums[n]
        return 0

    def insert(self, ngram):
        assert isinstance(ngram, list)
        if ngram:
            node = self._root
            last_word = ngram[-1]

            # Increment the vocabulary size count if new word added to root.
            if not self._root.has_child(ngram[0]):
                self._vocab_size += 1

            for word in ngram[:-1]:
                if not node.has_child(word):
                    node.add_child(word)
                node = node.get_child(word)

            if not node.has_child(last_word):
                node.add_child(last_word)
            node.get_child(last_word).inc()

            # Increment the ngram count for n.
            self._increment_total_count(len(ngram))

    def _get_node(self, word_seq):
        assert isinstance(word_seq, list)
        node = self._root

        for word in word_seq:
            if not node.has_child(word):
                return None
            node = node.get_child(word)

        return node

    def _increment_total_count(self, n):
        if not n in self._total_sums:
            self._total_sums[n] = 0
        self._total_sums[n] += 1

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