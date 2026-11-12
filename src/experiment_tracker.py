"""Append-only JSONL experiment tracker with atomic record writes and summaries."""
import json
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean
import uuid


class ExperimentTracker:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, experiment, metrics, config=None):
        if not experiment or not isinstance(experiment, str):
            raise ValueError("experiment must be a non-empty string")
        if not metrics or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in metrics.values()):
            raise ValueError("metrics must contain numerical values")
        record = {"id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat(),
                  "experiment": experiment, "config": config or {}, "metrics": metrics}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        return record

    def records(self, experiment=None):
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        return [r for r in records if experiment is None or r["experiment"] == experiment]

    def summarize(self, experiment, metric):
        values = [r["metrics"][metric] for r in self.records(experiment) if metric in r["metrics"]]
        if not values:
            raise ValueError("no matching metrics recorded")
        return {"experiment": experiment, "metric": metric, "runs": len(values),
                "mean": mean(values), "minimum": min(values), "maximum": max(values)}


def main():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as directory:
        tracker = ExperimentTracker(Path(directory) / "runs.jsonl")
        for seed, accuracy in [(1, .78), (2, .81), (3, .79)]:
            tracker.log("synthetic-classifier", {"accuracy": accuracy}, {"seed": seed})
        print(json.dumps(tracker.summarize("synthetic-classifier", "accuracy"), indent=2))


if __name__ == "__main__":
    main()
