"""Metrics computation for evaluation results."""

from collections import defaultdict
from typing import Optional

from src.core.types import Prediction, ResultSet


def compute_accuracy(predictions: list[Prediction]) -> float:
    """Compute overall accuracy."""
    if not predictions:
        return 0.0
    correct = sum(1 for p in predictions if p.correct)
    return correct / len(predictions)


def compute_category_accuracy(predictions: list[Prediction],
                               field: str) -> dict[str, dict]:
    """Group predictions by a metadata field and compute per-group accuracy.

    Returns:
        dict mapping category_value -> {"accuracy": float, "count": int, "correct": int}
    """
    groups = defaultdict(lambda: {"correct": 0, "total": 0})

    for p in predictions:
        value = p.metadata.get(field, "unknown")
        if value is None:
            value = "unknown"
        groups[value]["total"] += 1
        if p.correct:
            groups[value]["correct"] += 1

    result = {}
    for value, stats in sorted(groups.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        result[str(value)] = {
            "accuracy": round(acc, 4),
            "count": stats["total"],
            "correct": stats["correct"],
        }
    return result


def compute_delta(result_set: ResultSet,
                  baseline_set: Optional[ResultSet]) -> Optional[float]:
    """Compute accuracy delta vs baseline."""
    if baseline_set is None:
        return None
    return round(result_set.accuracy - baseline_set.accuracy, 4)


def compute_error_rate(predictions: list[Prediction]) -> float:
    """Compute error rate (ratio of failed inferences)."""
    if not predictions:
        return 0.0
    errors = sum(1 for p in predictions if p.error is not None)
    return errors / len(predictions)
