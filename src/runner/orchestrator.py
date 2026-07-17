"""Orchestrator: manages the full evaluation grid (models x benchmarks x perturbations).

Key design decisions:
- Outer loop on models: load once, run ALL benchmark x perturbation tasks, then unload.
  This minimizes expensive model load/unload cycles.
- Middle loop on benchmarks: load once per model, cache in memory.
- Inner loop on perturbations: iterate over conditions, skip completed tasks.
- Writes experiment metadata.json on completion.
"""

import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch

from src.core.config import (
    load_config,
    load_model_config,
    load_benchmark_config,
    resolve_perturbations,
)
from src.core.registry import get_model, get_benchmark, get_perturbation, list_models, list_benchmarks, list_perturbations
from src.core.types import ExperimentConfig
from src.data.base import AbstractBenchmark
from src.models.base import AbstractModel
from src.perturbations.base import Perturbation
from src.runner.task import EvaluationTask, ResultSet
from src.runner.logging_setup import setup_logging

# Trigger all registrations by importing subpackages
import src.data.mmau      # noqa: F401
import src.data.mmar      # noqa: F401
import src.models.qwen_omni     # noqa: F401
import src.models.qwen2_audio   # noqa: F401
import src.models.moss_audio    # noqa: F401
import src.models.kimi_audio    # noqa: F401
import src.models.mock_model    # noqa: F401
import src.perturbations.baseline          # noqa: F401
import src.perturbations.silent_audio      # noqa: F401
import src.perturbations.noise_audio       # noqa: F401
import src.perturbations.shuffled_choices  # noqa: F401
import src.perturbations.label_only        # noqa: F401


class Orchestrator:
    """Manages the full experiment life cycle."""

    def __init__(self, experiment_config: ExperimentConfig):
        self._config = experiment_config
        self._root_logger = setup_logging(level=experiment_config.runtime.log_level)
        self.output_dir: str = ""  # Set after run() creates the output directory

        # Set global seed
        import random
        random.seed(experiment_config.seed)

    def run(self) -> dict[str, ResultSet]:
        """Execute the full experiment grid.

        Returns:
            Dict mapping task_id -> ResultSet for all completed tasks.
        """
        output_dir = self._make_output_dir()
        self.output_dir = output_dir
        self._root_logger.info(f"Output directory: {output_dir}")
        self._root_logger.info(f"Models: {self._config.models}")
        self._root_logger.info(f"Benchmarks: {self._config.benchmarks}")
        self._root_logger.info(f"Perturbations: {self._config.perturbations}")

        all_results: dict[str, ResultSet] = {}
        experiment_started_at = time.time()
        failed_tasks: list[str] = []

        # Timing bookkeeping
        timing_model_load: dict[str, float] = {}       # model_name -> load seconds
        timing_benchmark_load: dict[str, float] = {}   # benchmark_name -> load seconds
        timing_per_task: dict[str, dict] = {}           # task_id -> breakdown dict

        # Resolve perturbations
        pert_configs = resolve_perturbations(self._config.perturbations)

        # Perturbations are independent of model audio support now
        # (label_only is an output-format constraint, keeps audio)

        # ── Outer loop: models ──
        for model_name in self._config.models:
            model_cfg = load_model_config(model_name)
            model_cls = get_model(model_name)
            model_cfg_dict = model_cfg.model_dump()
            model_cfg_dict["path"] = model_cfg.path  # include resolved @property
            model: AbstractModel = model_cls(config=model_cfg_dict)

            self._root_logger.info(f"\n{'='*60}\nLoading model: {model_name} ({model_cfg.display_name})\n{'='*60}")
            model_load_start = time.time()
            try:
                model.load()
            except Exception as e:
                self._root_logger.error(f"Failed to load {model_name}: {e}")
                failed_tasks.append(f"{model_name}__load_failed")
                continue
            model_load_elapsed = time.time() - model_load_start
            timing_model_load[model_name] = round(model_load_elapsed, 3)
            self._root_logger.info(f"Model loaded in {model_load_elapsed:.1f}s")

            # ── Middle loop: benchmarks ──
            for benchmark_name in self._config.benchmarks:
                bench_cfg = load_benchmark_config(benchmark_name)
                bench_cls = get_benchmark(benchmark_name)
                benchmark: AbstractBenchmark = bench_cls(
                    hf_dataset=bench_cfg.hf_dataset,
                    split=bench_cfg.split,
                    max_samples=bench_cfg.max_samples,
                    audio_root=bench_cfg.audio_root,
                )

                self._root_logger.info(f"Loading benchmark: {benchmark_name} ({bench_cfg.display_name})")
                bm_load_start = time.time()
                benchmark.load()
                bm_load_elapsed = time.time() - bm_load_start
                # Only record first load (subsequent are cached/no-op)
                if benchmark_name not in timing_benchmark_load:
                    timing_benchmark_load[benchmark_name] = round(bm_load_elapsed, 3)
                self._root_logger.info(f"Benchmark loaded in {bm_load_elapsed:.1f}s ({len(benchmark)} samples)")

                # ── Inner loop: perturbations ──
                for pc in pert_configs:
                    pert_cls = get_perturbation(pc.name)
                    perturbation: Perturbation = pert_cls()

                    # Initialize perturbation context (e.g., audio pool for unrelated)
                    if hasattr(perturbation, 'init_context'):
                        perturbation.init_context(benchmark)

                    task_id = self._make_task_id(model_name, benchmark_name, pc)
                    pred_file = os.path.join(output_dir, "predictions", f"{task_id}.jsonl")

                    # Skip if already completed
                    if self._config.runtime.resume and os.path.exists(pred_file):
                        self._root_logger.info(f"  [SKIP] {task_id} (already exists)")
                        continue

                    self._root_logger.info(f"  Running: {task_id}")

                    try:
                        task = EvaluationTask(
                            model=model,
                            benchmark=benchmark,
                            perturbation=perturbation,
                            perturbation_config=pc,
                            output_dir=output_dir,
                            seed=self._config.seed,
                            batch_size=self._config.runtime.batch_size,
                        )
                        task_start = time.time()
                        result = task.run(skip_completed=self._config.runtime.resume)
                        task_elapsed = time.time() - task_start
                        all_results[task_id] = result

                        # Record per-task timing breakdown
                        timing_per_task[task_id] = {
                            "duration_seconds": round(task_elapsed, 3),
                            "samples_per_second": result.samples_per_second,
                            "perturbation_time_seconds": result.perturbation_time_seconds,
                            "inference_time_seconds": result.inference_time_seconds,
                            "total_samples": result.total,
                            "errors": result.errors,
                            "accuracy": round(result.accuracy, 4),
                        }
                        self._root_logger.info(
                            f"    Done: {task_elapsed:.1f}s, "
                            f"{result.samples_per_second:.2f} sp/s, "
                            f"acc={result.accuracy:.4f}"
                        )
                    except Exception as e:
                        self._root_logger.error(f"  [FAILED] {task_id}: {e}")
                        failed_tasks.append(task_id)

                    # Clear GPU cache between tasks to avoid fragmentation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            # Unload model after all benchmark x perturbation tasks
            self._root_logger.info(f"Unloading model: {model_name}")
            model.unload()

        total_elapsed = time.time() - experiment_started_at

        # ── Aggregate timing summary ──
        total_pert_time = sum(t.get("perturbation_time_seconds", 0) for t in timing_per_task.values())
        total_infer_time = sum(t.get("inference_time_seconds", 0) for t in timing_per_task.values())

        # ── Write metadata ──
        metadata = {
            "experiment_name": self._config.name,
            "description": self._config.description,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seed": self._config.seed,
            "models": self._config.models,
            "benchmarks": self._config.benchmarks,
            "perturbations": self._config.perturbations,
            "total_tasks": len(self._config.models) * len(self._config.benchmarks) * len(pert_configs),
            "completed_tasks": len(all_results),
            "failed_tasks": failed_tasks,
            "total_samples_processed": sum(r.total for r in all_results.values()),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "timing": {
                "total_wall_seconds": round(total_elapsed, 1),
                "model_load_seconds": timing_model_load,
                "total_model_load_seconds": round(sum(timing_model_load.values()), 1),
                "benchmark_load_seconds": timing_benchmark_load,
                "total_benchmark_load_seconds": round(sum(timing_benchmark_load.values()), 1),
                "total_perturbation_seconds": round(total_pert_time, 1),
                "total_inference_seconds": round(total_infer_time, 1),
                "per_task": timing_per_task,
            },
        }

        meta_path = os.path.join(output_dir, "metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        self._root_logger.info(f"\n{'='*60}")
        self._root_logger.info(f"Experiment complete: {self._config.name}")
        self._root_logger.info(f"Completed: {len(all_results)} tasks, Failed: {len(failed_tasks)}")
        self._root_logger.info(f"Total wall time: {total_elapsed:.1f}s "
                               f"({total_elapsed/60:.1f}m, {total_elapsed/3600:.2f}h)")
        self._root_logger.info(f"Model loading: {sum(timing_model_load.values()):.1f}s | "
                               f"Benchmark loading: {sum(timing_benchmark_load.values()):.1f}s | "
                               f"Inference: {total_infer_time:.1f}s | "
                               f"Perturbation: {total_pert_time:.1f}s")
        self._root_logger.info(f"Metadata: {meta_path}")

        # ── Move complete failures to a separate directory ──
        if len(all_results) == 0:
            failed_base = os.path.join(self._config.output_dir, "failed")
            os.makedirs(failed_base, exist_ok=True)
            dest = os.path.join(failed_base, os.path.basename(output_dir))
            shutil.move(output_dir, dest)
            self.output_dir = dest
            output_dir = dest
            self._root_logger.info(f"Moved failed experiment to: {dest}")

        return all_results

    def _make_output_dir(self) -> str:
        """Create a timestamped output directory."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dirname = f"{self._config.name}_{timestamp}"
        output_dir = os.path.join(self._config.output_dir, dirname)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "predictions"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
        return output_dir

    @staticmethod
    def _make_task_id(model: str, benchmark: str, pc) -> str:
        return f"{model}__{benchmark}__{pc.name}"
