#!/usr/bin/env python3
"""Run a full experiment from a config file.

Usage:
    python scripts/run_experiment.py <experiment_name> [--models M1,M2] [--list-models]

Examples:
    python scripts/run_experiment.py mmau_full                          # all models
    python scripts/run_experiment.py mmau_full --models qwen_omni       # single model
    python scripts/run_experiment.py mmau_full -m qwen_omni -m kimi_audio  # two models
    python scripts/run_experiment.py mmau_full --list-models             # show available models
"""

import argparse
import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_config
from src.runner.orchestrator import Orchestrator
from src.evaluation.reporter import aggregate_summary, aggregate_by_category, build_delta_matrix
from src.evaluation.exporter import (
    write_summary_csv,
    write_category_csv,
    write_delta_csv,
    write_markdown_report,
)


def list_experiments():
    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "experiments")
    for f in sorted(os.listdir(config_dir)):
        if f.endswith(".yaml"):
            print(f"  {f.replace('.yaml', '')}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a full experiment from a config file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  %(prog)s mmau_full                          # all models\n"
            "  %(prog)s mmau_full --models qwen_omni       # single model\n"
            "  %(prog)s mmau_full -m qwen_omni -m kimi_audio  # two models\n"
            "  %(prog)s mmau_full --list-models             # show available models"
        ),
    )
    parser.add_argument("experiment", nargs="?", help="Experiment config name (without .yaml)")
    parser.add_argument(
        "--models", "-m",
        action="append",
        dest="models",
        help="Only run these model(s). Repeat for multiple: -m qwen_omni -m kimi_audio. "
             "Comma-separated also works: --models qwen_omni,kimi_audio",
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List models in the experiment and exit",
    )
    args = parser.parse_args()

    if not args.experiment:
        print("Usage: python scripts/run_experiment.py <experiment_name> [--models ...]")
        print("Available experiments:")
        list_experiments()
        sys.exit(1)

    experiment_name = args.experiment
    print(f"Loading experiment: {experiment_name}")

    # Load config
    config = load_config(experiment_name)
    print(f"Description: {config.description}")

    # Filter models if --models specified
    if args.models:
        # Support both comma-separated and repeated flags
        requested = []
        for item in args.models:
            requested.extend(m.strip() for m in item.split(",") if m.strip())

        # Validate
        valid = set(config.models)
        invalid = [m for m in requested if m not in valid]
        if invalid:
            print(f"Error: model(s) not in experiment '{experiment_name}': {', '.join(invalid)}")
            print(f"Available models: {', '.join(sorted(valid))}")
            sys.exit(1)

        config.models = requested
        print(f"Models (filtered): {', '.join(requested)}")
    else:
        print(f"Models: {', '.join(config.models)}")

    if args.list_models:
        print(f"\nAvailable models in '{experiment_name}':")
        for m in config.models:
            print(f"  - {m}")
        sys.exit(0)

    # Run orchestrator
    orch = Orchestrator(config)
    all_results = orch.run()

    # Generate reports
    output_dir = orch.output_dir
    if not output_dir:
        print("No output directory set by orchestrator.")
        return

    reports_dir = os.path.join(output_dir, "reports")

    # Category fields per benchmark
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

    print(f"\nReports written to: {reports_dir}")
    print("  - summary.csv")
    print("  - by_category.csv")
    print("  - delta_vs_baseline.csv")
    print("  - report.md")

    # Print quick summary
    print("\nQuick summary:")
    for row in summary:
        delta_str = f" (delta: {row['delta_vs_baseline']:+.3f})" if row['delta_vs_baseline'] is not None else ""
        print(f"  {row['model']:20s} | {row['benchmark']:6s} | {row['perturbation']:25s} | "
              f"acc={row['accuracy']:.4f}{delta_str}")


if __name__ == "__main__":
    main()
