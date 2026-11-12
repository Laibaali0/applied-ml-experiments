import math
import pytest
from src.semantic_search import DocumentSearch
from src.experiment_tracker import ExperimentTracker
from src.recommender import ItemRecommender


def test_search_relevance():
    engine = DocumentSearch(["GPU model quantization", "Fraud risk model", "GPU inference speed"])
    found = engine.search("GPU inference", k=1)
    assert found[0][0] == "GPU inference speed"
    assert found[0][1] > 0


def test_search_invalid_k():
    with pytest.raises(ValueError):
        DocumentSearch(["hello"]).search("hello", k=2)


def test_tracker_roundtrip(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.jsonl")
    tracker.log("model", {"accuracy": .8}, {"seed": 1})
    tracker.log("model", {"accuracy": .9}, {"seed": 2})
    result = tracker.summarize("model", "accuracy")
    assert result["runs"] == 2
    assert math.isclose(result["mean"], .85)
    assert len(tracker.records()) == 2


def test_tracker_missing_metric(tmp_path):
    tracker = ExperimentTracker(tmp_path / "runs.jsonl")
    with pytest.raises(ValueError):
        tracker.summarize("no-experiment", "accuracy")


def test_recommender_filters_known_items():
    r = ItemRecommender([("a", "shoes", 5), ("b", "shoes", 4), ("b", "bag", 5), ("c", "bag", 5)])
    recommendations = r.recommend("a")
    assert recommendations and recommendations[0][0] == "bag"
    assert r.recommend("new") == []


def test_invalid_rating():
    with pytest.raises(ValueError):
        ItemRecommender([("user", "item", -1)])
