"""Reproducible item-based recommendation using explicit ratings and cosine similarity."""
from collections import defaultdict
import math


class ItemRecommender:
    def __init__(self, ratings):
        # ratings: [(user, item, score), ...]; scores must be positive
        self.user_items = defaultdict(dict)
        self.item_users = defaultdict(dict)
        for user, item, score in ratings:
            if score <= 0:
                raise ValueError("rating scores must be positive")
            self.user_items[user][item] = score
            self.item_users[item][user] = score

    def _similarity(self, first, second):
        a, b = self.item_users[first], self.item_users[second]
        common = a.keys() & b.keys()
        if not common:
            return 0.0
        numerator = sum(a[user] * b[user] for user in common)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        return numerator / (norm_a * norm_b) if norm_a and norm_b else 0.0

    def recommend(self, user, k=3):
        if k < 1:
            raise ValueError("k must be positive")
        known = self.user_items.get(user, {})
        if not known:
            return []
        scores = []
        for candidate in self.item_users:
            if candidate in known:
                continue
            weights = [(self._similarity(item, candidate), rating) for item, rating in known.items()]
            numerator = sum(weight * rating for weight, rating in weights)
            denominator = sum(weight for weight, _ in weights)
            if denominator:
                scores.append((candidate, numerator / denominator))
        return sorted(scores, key=lambda pair: (-pair[1], pair[0]))[:k]


def main():
    ratings = [("u1", "shoes", 5), ("u1", "jacket", 3),
               ("u2", "shoes", 4), ("u2", "bag", 5),
               ("u3", "jacket", 4), ("u3", "bag", 4),
               ("u4", "bag", 4), ("u4", "shoes", 4)]
    print("Recommended for u1:", ItemRecommender(ratings).recommend("u1"))


if __name__ == "__main__":
    main()
