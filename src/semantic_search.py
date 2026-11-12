"""Tiny TF-IDF based semantic-style document retrieval demo (no neural model)."""
from collections import Counter
import math
import re


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class DocumentSearch:
    def __init__(self, documents):
        self.docs = list(documents)
        if not self.docs:
            raise ValueError("Provide at least one document")
        self.tokens = [tokenize(doc) for doc in self.docs]
        self.df = Counter(token for tokens in self.tokens for token in set(tokens))
        self.n = len(self.docs)
        self.vectors = [self._vector(tokens) for tokens in self.tokens]

    def _vector(self, tokens):
        counts = Counter(tokens)
        return {term: (1 + math.log(count)) * (1 + math.log((self.n + 1) / (self.df.get(term, 0) + 1)))
                for term, count in counts.items()}

    @staticmethod
    def _cosine(a, b):
        numer = sum(value * b.get(key, 0) for key, value in a.items())
        na = math.sqrt(sum(value * value for value in a.values()))
        nb = math.sqrt(sum(value * value for value in b.values()))
        return numer / (na * nb) if na and nb else 0.0

    def search(self, query, k=3):
        if not 1 <= k <= len(self.docs):
            raise ValueError("k must fit the corpus size")
        q = self._vector(tokenize(query))
        ranked = sorted(enumerate(self.vectors), key=lambda pair: (-self._cosine(q, pair[1]), pair[0]))
        return [(self.docs[i], round(self._cosine(q, vec), 4)) for i, vec in ranked[:k]]


def main():
    corpus = [
        "How to optimize transformer inference latency and memory",
        "Evaluating fraud detection using a temporal holdout",
        "Building an offline reinforcement learning recommendation system",
        "Quantization for large language models on GPU hardware",
    ]
    for doc, score in DocumentSearch(corpus).search("transformer GPU inference", k=3):
        print(f"{score:.3f} | {doc}")


if __name__ == "__main__":
    main()
