#!/usr/bin/env python3
"""Re-aggregate results from an existing experiment output directory.

Usage:
    python scripts/aggregate.py outputs/mmau_robustness_v1_20260709_143000
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.types import Prediction, ResultSet
from src.evaluation.reporter import aggregate_summary, aggregate_by_category, build_delta_matrix
from src.evaluation.exporter import (
    write_summary_csv,
    write_category_csv,
    write_delta_csv,
    write_markdown_report,
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/aggregate.py <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    pred_dir = os.path.join(output_dir, "predictions")

    if not os.path.isdir(pred_dir):
        print(f"Predictions directory not found: {pred_dir}")
        sys.exit(1)

    # Load all prediction files
    all_results: dict[str, ResultSet] = {}

    for fname in sorted(os.listdir(pred_dir)):
        if not fname.endswith(".jsonl"):
            continue

        filepath = os.path.join(pred_dir, fname)
        predictions: list[Prediction] = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    predictions.append(Prediction(
                        sample_id=data["sample_id"],
                        question=data.get("question", ""),
                        choices=data.get("choices", []),
                        ground_truth=data["ground_truth"],
                        chosen_answer=data["chosen_answer"],
                        raw_output=data.get("raw_output", ""),
                        correct=data["correct"],
                        error=data.get("error"),
                        metadata=data.get("metadata", {}),
                        perturbation=data.get("perturbation", ""),
                        model=data.get("model", ""),
                        benchmark=data.get("benchmark", ""),
                    ))
                except (json.JSONDecodeError, KeyError):
                    continue

        if predictions:
            # Extract task info from first prediction
            p = predictions[0]
            rs = ResultSet(
                model=p.model,
                benchmark=p.benchmark,
                perturbation=p.perturbation,
                predictions=predictions,
                errors=sum(1 for p in predictions if p.error),
                total=len(predictions),
            )
            task_id = fname.replace(".jsonl", "")
            all_results[task_id] = rs
            print(f"Loaded {task_id}: {len(predictions)} predictions, acc={rs.accuracy:.4f}")

    if not all_results:
        print("No prediction files found.")
        sys.exit(1)

    print(f"\nTotal: {len(all_results)} tasks loaded.")

    # Generate reports
    reports_dir = os.path.join(output_dir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    category_fields = {
        "mmau": ["dataset", "task", "category", "sub_category", "difficulty"],
        "mmar": ["modality", "category", "sub_category", "language", "source"],
    }

    summary = aggregate_summary(all_results)
    by_cat = aggregate_by_category(all_results, category_fields)
    delta = build_delta_matrix(all_results)

    write_summary_csv(summary, os.path.join(reports_dir, "summary.csv"))
    write_category_csv(by_cat, os.path.join(reports_dir, "by_category.csv"))
    write_delta_csv(delta, os.path.join(reports_dir, "delta_vs_baseline.csv"))
    # Load metadata for timing info in report
    metadata = None
    meta_path = os.path.join(output_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    write_markdown_report(summary, by_cat, delta, os.path.join(reports_dir, "report.md"),
                          metadata=metadata)

    print(f"Reports written to: {reports_dir}")
    print(f"  - summary.csv ({len(summary)} rows)")
    print(f"  - by_category.csv ({len(by_cat)} rows)")
    print(f"  - delta_vs_baseline.csv ({len(delta)} rows)")
    print(f"  - report.md")


if __name__ == "__main__":
    main()
