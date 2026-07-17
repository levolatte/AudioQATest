"""Exporter: writes results to CSV, Markdown, and JSON formats."""

import csv
import json
import os
from typing import Optional

from src.core.types import ResultSet


def write_summary_csv(rows: list[dict], path: str) -> None:
    """Write summary rows to CSV."""
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_category_csv(rows: list[dict], path: str) -> None:
    """Write category breakdown rows to CSV."""
    write_summary_csv(rows, path)


def write_delta_csv(rows: list[dict], path: str) -> None:
    """Write delta matrix rows to CSV."""
    write_summary_csv(rows, path)


def write_markdown_report(summary_rows: list[dict],
                          category_rows: list[dict],
                          delta_rows: list[dict],
                          path: str,
                          metadata: dict = None) -> None:
    """Generate a human-readable Markdown report.

    Args:
        summary_rows: Overall accuracy rows from aggregate_summary().
        category_rows: Per-category breakdown from aggregate_by_category().
        delta_rows: Robustness delta matrix from build_delta_matrix().
        path: Output file path.
        metadata: Optional experiment metadata dict (includes timing info).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lines = []
    lines.append("# Audio QA Robustness Evaluation Report\n")

    # ── Experiment overview ──
    if metadata:
        timing = metadata.get("timing", {})
        lines.append("## Experiment Overview\n")
        lines.append(f"- **Experiment**: {metadata.get('experiment_name', 'N/A')}")
        lines.append(f"- **Description**: {metadata.get('description', 'N/A')}")
        lines.append(f"- **Timestamp**: {metadata.get('timestamp', 'N/A')}")
        lines.append(f"- **Device**: {metadata.get('device', 'N/A')}")
        lines.append(f"- **Total wall time**: {_fmt_duration(timing.get('total_wall_seconds', 0))}")
        lines.append(f"- **Models**: {', '.join(metadata.get('models', []))}")
        lines.append(f"- **Benchmarks**: {', '.join(metadata.get('benchmarks', []))}")
        lines.append(f"- **Perturbations**: {', '.join(metadata.get('perturbations', []))}")
        lines.append("")

    # ── Timing summary ──
    if metadata and metadata.get("timing"):
        timing = metadata["timing"]
        lines.append("## Timing Breakdown\n")
        lines.append("| Phase | Duration (s) | Duration (min) |")
        lines.append("| --- | --- | --- |")
        total = timing.get("total_wall_seconds", 0)
        model_load = timing.get("total_model_load_seconds", 0)
        bm_load = timing.get("total_benchmark_load_seconds", 0)
        perturb = timing.get("total_perturbation_seconds", 0)
        inference = timing.get("total_inference_seconds", 0)
        overhead = total - model_load - bm_load - perturb - inference
        lines.append(f"| Total | {total:.1f} | {total/60:.1f} |")
        lines.append(f"| Model loading | {model_load:.1f} | {model_load/60:.1f} |")
        lines.append(f"| Benchmark loading | {bm_load:.1f} | {bm_load/60:.1f} |")
        lines.append(f"| Perturbation | {perturb:.1f} | {perturb/60:.1f} |")
        lines.append(f"| Inference | {inference:.1f} | {inference/60:.1f} |")
        lines.append(f"| Overhead (I/O, logging) | {overhead:.1f} | {overhead/60:.1f} |")
        lines.append("")

        # Per-task timing table
        per_task = timing.get("per_task", {})
        if per_task:
            lines.append("### Per-Task Timing\n")
            lines.append("| Task | Duration (s) | Samples/s | Accuracy |")
            lines.append("| --- | --- | --- | --- |")
            for task_id, tt in sorted(per_task.items()):
                lines.append(
                    f"| {task_id} | {tt.get('duration_seconds', 0):.1f} | "
                    f"{tt.get('samples_per_second', 0):.2f} | "
                    f"{tt.get('accuracy', 0):.4f} |"
                )
            lines.append("")

    # ── Overall summary table ──
    lines.append("## Overall Accuracy\n")
    if summary_rows:
        # Separate timing columns from main summary for readability
        main_headers = [h for h in summary_rows[0].keys()
                        if h not in ("duration_s", "samples_per_second")]
        timing_headers = [h for h in ["duration_s", "samples_per_second"]
                          if h in summary_rows[0]]
        lines.append("| " + " | ".join(main_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(main_headers)) + " |")
        for row in summary_rows:
            vals = [str(row.get(h, "")) for h in main_headers]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

        if timing_headers:
            lines.append("### Per-Task Throughput\n")
            lines.append("| model | benchmark | perturbation | duration_s | samples/s |")
            lines.append("| --- | --- | --- | --- | --- |")
            for row in summary_rows:
                lines.append(
                    f"| {row['model']} | {row['benchmark']} | {row['perturbation']} | "
                    f"{row.get('duration_s', '-')} | {row.get('samples_per_second', '-')} |"
                )
            lines.append("")

    # ── Robustness delta matrix ──
    lines.append("## Robustness Gap (Accuracy drop from Baseline)\n")
    if delta_rows:
        headers = list(delta_rows[0].keys())
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in delta_rows:
            vals = [str(row.get(h, "")) for h in headers]
            lines.append("| " + " | ".join(vals) + " |")
        lines.append("")

    # ── Category breakdown ──
    lines.append("## Per-Category Breakdown\n")
    if category_rows:
        # Group by model, benchmark, perturbation
        current_header = None
        for row in category_rows:
            header = f"### {row['model']} / {row['benchmark']} / {row['perturbation']}\n"
            if header != current_header:
                current_header = header
                lines.append(header)
                lines.append("| category_field | category_value | accuracy | count | correct |")
                lines.append("| --- | --- | --- | --- | --- |")
            lines.append(
                f"| {row['category_field']} | {row['category_value']} | "
                f"{row['accuracy']} | {row['count']} | {row['correct']} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m ({seconds:.0f}s)"
    else:
        return f"{seconds/3600:.2f}h ({seconds/60:.1f}m)"


def write_metadata_json(metadata: dict, path: str) -> None:
    """Write experiment metadata to JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
