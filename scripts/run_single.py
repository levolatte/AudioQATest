#!/usr/bin/env python3
"""Run a single model/benchmark/perturbation combination.

Usage:
    python scripts/run_single.py <model> <benchmark> <perturbation> [--max-samples N]
    python scripts/run_single.py qwen_omni mmau baseline --max-samples 10
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import load_model_config, load_benchmark_config, resolve_perturbations
from src.core.registry import get_model, get_benchmark, get_perturbation
from src.runner.task import EvaluationTask


def main():
    parser = argparse.ArgumentParser(description="Run single evaluation task")
    parser.add_argument("model", help="Model name (e.g., qwen_omni)")
    parser.add_argument("benchmark", help="Benchmark name (e.g., mmau)")
    parser.add_argument("perturbation", help="Perturbation name (e.g., baseline, silent_audio)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for inference (default: 24)")
    parser.add_argument("--output-dir", default="outputs/single_run", help="Output directory")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip existing predictions")
    args = parser.parse_args()

    # Load configs
    model_cfg = load_model_config(args.model)
    bench_cfg = load_benchmark_config(args.benchmark)
    if args.max_samples:
        bench_cfg.max_samples = args.max_samples

    # Resolve perturbation
    pert_configs = resolve_perturbations([args.perturbation])
    pc = pert_configs[0]

    # Instantiate
    model_cls = get_model(args.model)
    model_cfg_dict = model_cfg.model_dump()
    model_cfg_dict["path"] = model_cfg.path  # include resolved @property
    model = model_cls(config=model_cfg_dict)

    bench_cls = get_benchmark(args.benchmark)
    benchmark = bench_cls(
        hf_dataset=bench_cfg.hf_dataset,
        split=bench_cfg.split,
        max_samples=bench_cfg.max_samples,
        audio_root=bench_cfg.audio_root,
    )

    pert_cls = get_perturbation(pc.name)
    perturbation = pert_cls()
    # Init context needs the benchmark to be loaded (for lazy audio access).
    # We load here so init_context works, then model.load() follows.
    benchmark.load()
    if hasattr(perturbation, 'init_context'):
        try:
            perturbation.init_context(benchmark)
        except Exception:
            pass

    print(f"Model:      {args.model} ({model_cfg.display_name})")
    print(f"Benchmark:  {args.benchmark} ({bench_cfg.display_name})")
    print(f"Perturbation: {pc.name} {pc.params}")
    print(f"Max samples: {bench_cfg.max_samples or 'all'}")

    # Load and run
    print("\nLoading model...")
    model.load()

    print("Running...")
    task = EvaluationTask(
        model=model,
        benchmark=benchmark,
        perturbation=perturbation,
        perturbation_config=pc,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
    result = task.run(skip_completed=not args.no_resume)

    print(f"\nDone. Accuracy: {result.accuracy:.4f} "
          f"({result.correct_count}/{len(benchmark)}), errors: {result.errors}")
    print(f"Duration: {result.duration_seconds:.1f}s | "
          f"Throughput: {result.samples_per_second:.2f} samples/s")
    print(f"Breakdown — Perturbation: {result.perturbation_time_seconds:.1f}s | "
          f"Inference: {result.inference_time_seconds:.1f}s")

    model.unload()


if __name__ == "__main__":
    main()
