# Applied ML Experiments

Three small, runnable, dependency-light engineering projects with tests. All examples use **synthetic or hand-authored data**, are for learning/portfolio development, and are not representations of employer work or production deployments.

## Projects

| Project | Concepts | Run |
| --- | --- | --- |
| [Document Search](src/semantic_search.py) | Tokenization, TF-IDF, cosine similarity, ranking | `python -m src.semantic_search` |
| [Experiment Tracker](src/experiment_tracker.py) | JSONL logging, experiment settings, multi-run metrics | `python -m src.experiment_tracker` |
| [Item Recommender](src/recommender.py) | Item similarity, user-item interactions, ranking | `python -m src.recommender` |

## Quick start

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.semantic_search
python -m src.experiment_tracker
python -m src.recommender
```

## How to extend

- Document Search: evaluate Precision@K and MRR on a labeled public corpus; add a neural embedding reranker.
- Experiment Tracker: add CSV export, experiment comparison, and safe concurrent writing.
- Item Recommender: measure Recall@K and NDCG on held-out user interactions, compare with popularity recommendations.

**Limitations:** This is educational code. The search is sparse lexical retrieval, not neural semantic search; the recommender is a toy collaborative filtering baseline, not a large-scale recommendation model. No real-user metrics or deployment claims are made.
