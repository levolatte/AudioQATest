"""Reporter: aggregates across multiple ResultSets and generates summary data."""

from collections import defaultdict
from typing import Optional

from src.core.types import Prediction, ResultSet
from src.evaluation.metrics import compute_accuracy, compute_category_accuracy, compute_delta


def aggregate_summary(all_results: dict[str, ResultSet]) -> list[dict]:
    """Generate a flat summary table: one row per (model, benchmark, perturbation).

    Returns:
        List of dicts with keys: model, benchmark, perturbation, accuracy,
        total, correct, errors, delta_vs_baseline.
    """
    # Find baseline results for delta computation
    baselines: dict[tuple[str, str], float] = {}
    for task_id, rs in all_results.items():
        if "baseline" in task_id:
            key = (rs.model, rs.benchmark)
            baselines[key] = rs.accuracy

    rows = []
    for task_id, rs in all_results.items():
        key = (rs.model, rs.benchmark)
        baseline_acc = baselines.get(key, None)
        delta = round(rs.accuracy - baseline_acc, 4) if baseline_acc is not None else None

        rows.append({
            "model": rs.model,
            "benchmark": rs.benchmark,
            "perturbation": rs.perturbation,
            "accuracy": round(rs.accuracy, 4),
            "total": rs.total,
            "correct": rs.correct_count,
            "errors": rs.errors,
            "delta_vs_baseline": delta,
            "duration_s": rs.duration_seconds,
            "samples_per_second": rs.samples_per_second,
        })

    return sorted(rows, key=lambda r: (r["model"], r["benchmark"], r["perturbation"]))


def aggregate_by_category(all_results: dict[str, ResultSet],
                          category_fields: dict[str, list[str]]) -> list[dict]:
    """Generate category-level breakdown rows.

    Args:
        all_results: Map of task_id -> ResultSet.
        category_fields: Dict mapping benchmark_name -> list of metadata fields
                         to break down by (from benchmark.category_fields).

    Returns:
        List of dicts with keys: model, benchmark, perturbation,
        category_field, category_value, accuracy, count, correct.
    """
    rows = []
    for task_id, rs in all_results.items():
        fields = category_fields.get(rs.benchmark, [])
        for field in fields:
            breakdown = compute_category_accuracy(rs.predictions, field)
            for value, stats in breakdown.items():
                rows.append({
                    "model": rs.model,
                    "benchmark": rs.benchmark,
                    "perturbation": rs.perturbation,
                    "category_field": field,
                    "category_value": value,
                    "accuracy": stats["accuracy"],
                    "count": stats["count"],
                    "correct": stats["correct"],
                })

    return sorted(rows, key=lambda r: (r["model"], r["benchmark"],
                                        r["perturbation"], r["category_field"],
                                        r["category_value"]))


def build_delta_matrix(all_results: dict[str, ResultSet]) -> list[dict]:
    """Build a robustness gap matrix: accuracy drop from baseline per perturbation.

    Returns:
        List of dicts with keys: model, benchmark, and one column per perturbation.
    """
    # Group by model x benchmark
    groups = defaultdict(dict)
    for task_id, rs in all_results.items():
        groups[(rs.model, rs.benchmark)][rs.perturbation] = rs.accuracy

    rows = []
    for (model, benchmark), pert_accs in sorted(groups.items()):
        baseline_acc = pert_accs.get("baseline", 0)
        row = {"model": model, "benchmark": benchmark}
        for pert, acc in sorted(pert_accs.items()):
            if pert != "baseline":
                row[pert] = round(acc - baseline_acc, 4)
        rows.append(row)

    return rows
