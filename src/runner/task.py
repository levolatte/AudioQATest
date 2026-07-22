"""Evaluation task: handles a single (model, benchmark, perturbation) run."""

import json
import os
import random
import time
from typing import Optional, Set

import soundfile as sf
import torch
from tqdm import tqdm

from src.core.types import Sample, Prediction, ResultSet, PerturbationConfig, Sample
from src.data.base import AbstractBenchmark
from src.models.base import AbstractModel
from src.models.utils import check_strict_match
from src.perturbations.base import Perturbation
from src.runner.logging_setup import get_task_logger


class EvaluationTask:
    """Encapsulates one evaluation run: one model, one benchmark, one perturbation.

    Features:
    - Incremental JSONL output for crash resilience
    - Resume support: re-reads existing predictions and skips completed samples
    - Per-task logging
    """

    def __init__(self, model: AbstractModel, benchmark: AbstractBenchmark,
                 perturbation: Perturbation, perturbation_config: PerturbationConfig,
                 output_dir: str, seed: int = 42, batch_size: int = 1):
        self.model = model
        self.benchmark = benchmark
        self.perturbation = perturbation
        self.perturbation_config = perturbation_config
        self.output_dir = output_dir
        self.seed = seed
        self.batch_size = max(1, batch_size)
        self._rng = random.Random(seed)

    @property
    def perturbation_id(self) -> str:
        """Clean perturbation identifier.

        Examples: "baseline", "noise_audio", "silent_audio".
        """
        return self.perturbation_config.name

    @property
    def task_id(self) -> str:
        return f"{self.model.name}__{self.benchmark.name}__{self.perturbation_id}"

    def run(self, skip_completed: bool = True) -> ResultSet:
        """Execute the evaluation task, optionally with batched inference.

        Args:
            skip_completed: If True, skip samples that already have predictions.

        Returns:
            ResultSet with all predictions.
        """
        pred_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        pred_file = os.path.join(pred_dir, f"{self.task_id}.jsonl")

        logger = get_task_logger(
            model=self.model.name,
            benchmark=self.benchmark.name,
            perturbation=self.perturbation_id,
            log_dir=os.path.join(self.output_dir, "logs"),
        )

        logger.info(f"=== Task start: {self.task_id} ===")
        logger.info(f"Model: {self.model.name}, Benchmark: {self.benchmark.name}, "
                    f"Perturbation: {self.perturbation.name}, "
                    f"Samples: {len(self.benchmark)}, "
                    f"Batch size: {self.batch_size}")

        # Load existing predictions for resume
        completed_ids: Set[str] = set()
        predictions: list[Prediction] = []

        if skip_completed and os.path.exists(pred_file):
            with open(pred_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        pred = json.loads(line.strip())
                        completed_ids.add(pred["sample_id"])
                        predictions.append(Prediction(**pred))
                    except (json.JSONDecodeError, TypeError):
                        continue
            logger.info(f"Resume: {len(completed_ids)} already completed, "
                        f"{len(self.benchmark) - len(completed_ids)} remaining")

        n_resumed = len(completed_ids)
        errors = 0
        start_time = time.time()
        pert_time_total = 0.0
        infer_time_total = 0.0

        # Use tqdm with dynamic postfix
        pbar = tqdm(total=len(self.benchmark), desc=self.task_id, unit="samples")
        if n_resumed > 0:
            pbar.update(n_resumed)

        # ── Pre-compute audio durations for length-sorted batching ──
        # Sorting by duration minimises padding waste within each batch:
        # a 1 s sample won't be padded to 30 s just because it shares a
        # batch with a long utterance.
        #
        # Prefer metadata "audio_duration" (set by lazy-loading benchmarks
        # from the WAV header without decoding) so that sorting doesn't
        # force eager audio extraction.
        sample_order: list[tuple[int, float]] = []
        for i in range(len(self.benchmark)):
            if self.benchmark[i].id in completed_ids:
                continue
            sample = self.benchmark[i]
            dur = sample.metadata.get("audio_duration", None)
            if dur is None and sample.audio is not None and isinstance(sample.audio, str):
                try:
                    dur = sf.info(sample.audio).duration
                except Exception:
                    dur = 0.0
            if dur is None:
                dur = 0.0
            sample_order.append((i, dur))

        sample_order.sort(key=lambda x: x[1])

        if sample_order:
            logger.info(
                f"Sorted {len(sample_order)} samples by audio duration "
                f"(range: {sample_order[0][1]:.1f}s – {sample_order[-1][1]:.1f}s)"
            )

        # ── Main loop: accumulate samples into batches (duration-sorted) ──
        pending: list[tuple[int, Sample, Sample]] = []  # (index, original, transformed)

        for i, _dur in sample_order:
            sample = self.benchmark[i]

            # ── Apply perturbation ──
            pert_start = time.time()
            try:
                transformed = self.perturbation.apply(
                    sample, self._rng, **self.perturbation_config.params
                )
            except Exception as e:
                logger.error(f"Perturbation failed for {sample.id}: {e}")
                errors += 1
                pbar.update(1)
                continue
            pert_time_total += time.time() - pert_start

            pending.append((i, sample, transformed))

            # Flush batch when full
            if len(pending) >= self.batch_size:
                infer_time_total += self._flush_batch(
                    pending, predictions, pred_file, completed_ids, pbar, logger,
                )
                pending = []

        # Flush remainder
        if pending:
            infer_time_total += self._flush_batch(
                pending, predictions, pred_file, completed_ids, pbar, logger,
            )

        pbar.close()

        # ── Final timing summary ──
        elapsed = time.time() - start_time
        n_new = len(predictions) - n_resumed
        overall_sps = n_new / elapsed if elapsed > 0 else 0

        result_set = ResultSet(
            model=self.model.name,
            benchmark=self.benchmark.name,
            perturbation=self.perturbation_id,
            predictions=predictions,
            errors=errors,
            total=len(self.benchmark),
            duration_seconds=round(elapsed, 3),
            samples_per_second=round(overall_sps, 3),
            perturbation_time_seconds=round(pert_time_total, 3),
            inference_time_seconds=round(infer_time_total, 3),
        )

        logger.info(f"=== Task complete: {self.task_id} ===")
        logger.info(f"Accuracy: {result_set.accuracy:.4f} "
                    f"({result_set.correct_count}/{len(predictions)}, "
                    f"errors: {errors})")
        logger.info(f"Strict accuracy: {result_set.strict_accuracy:.4f} "
                    f"({result_set.strict_correct_count}/{len(predictions)})")
        logger.info(f"Duration: {elapsed:.1f}s ({n_new} new + {n_resumed} resumed) | "
                    f"Throughput: {overall_sps:.2f} samples/s")
        logger.info(f"Breakdown — Perturbation: {pert_time_total:.1f}s | "
                    f"Inference: {infer_time_total:.1f}s | "
                    f"Overhead: {elapsed - pert_time_total - infer_time_total:.1f}s")

        return result_set

    def _flush_batch(
        self,
        pending: list[tuple[int, Sample, Sample]],
        predictions: list[Prediction],
        pred_file: str,
        completed_ids: set,
        pbar,
        logger,
    ) -> float:
        """Run batch inference on pending samples and write predictions.

        Returns elapsed inference time (seconds).
        """
        # Build batch input list
        batch_inputs: list[tuple] = []
        for idx, sample, transformed in pending:
            batch_inputs.append((
                transformed.audio,
                transformed.question,
                transformed.choices,
                transformed.metadata.get("label_only", False),
            ))

        # ── Batch inference ──
        infer_start = time.time()
        batch_ok = True
        batch_results = []

        try:
            batch_results = self.model.infer_batch(batch_inputs)
        except Exception as e:
            logger.error(f"Batch inference failed ({len(pending)} samples): {e}")
            torch.cuda.empty_cache()
            batch_ok = False

        infer_elapsed = time.time() - infer_start

        # ── Process each result ──
        for j, (idx, sample, transformed) in enumerate(pending):
            if batch_ok and j < len(batch_results):
                chosen, raw = batch_results[j]
                correct = (chosen == transformed.ground_truth)
                strict_correct = (
                    check_strict_match(raw, transformed.ground_truth, transformed.choices)
                    if not transformed.metadata.get("label_only", False) else False
                )
                error_msg = None
            else:
                # Fallback: sequential per-sample inference
                try:
                    chosen, raw = self.model.infer(
                        transformed.audio, transformed.question, transformed.choices,
                        label_only=transformed.metadata.get("label_only", False),
                    )
                    correct = (chosen == transformed.ground_truth)
                    strict_correct = (
                        check_strict_match(raw, transformed.ground_truth, transformed.choices)
                        if not transformed.metadata.get("label_only", False) else False
                    )
                    error_msg = None
                except Exception as e2:
                    logger.error(f"Inference failed for {sample.id}: {e2}")
                    chosen = transformed.choices[0] if transformed.choices else ""
                    raw = f"[ERROR] {repr(e2)}"
                    correct = False
                    strict_correct = False
                    error_msg = repr(e2)
                    torch.cuda.empty_cache()

            prediction = Prediction(
                sample_id=sample.id,
                question=sample.question,
                choices=transformed.choices,
                ground_truth=transformed.ground_truth,
                chosen_answer=chosen,
                raw_output=raw,
                correct=correct,
                strict_correct=strict_correct,
                error=error_msg,
                metadata=transformed.metadata,
                perturbation=self.perturbation.name,
                model=self.model.name,
                benchmark=self.benchmark.name,
            )

            predictions.append(prediction)

            with open(pred_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(_pred_to_dict(prediction), ensure_ascii=False) + "\n")

            completed_ids.add(sample.id)
            pbar.update(1)

        # Update tqdm postfix with accuracy and batch size
        n_correct = sum(1 for p in predictions if p.correct)
        running_acc = n_correct / len(predictions) if predictions else 0
        pbar.set_postfix({"acc": f"{running_acc:.3f}", "bs": str(len(pending))})

        return infer_elapsed


def _pred_to_dict(p: Prediction) -> dict:
    """Convert a Prediction to a JSON-serializable dict."""
    d = {
        "sample_id": p.sample_id,
        "question": p.question,
        "choices": p.choices,
        "ground_truth": p.ground_truth,
        "chosen_answer": p.chosen_answer,
        "raw_output": p.raw_output,
        "correct": p.correct,
        "strict_correct": p.strict_correct,
        "error": p.error,
        "metadata": _serialize_metadata(p.metadata),
        "perturbation": p.perturbation,
        "model": p.model,
        "benchmark": p.benchmark,
    }
    return d


def _serialize_metadata(meta: dict) -> dict:
    """Convert metadata values to JSON-serializable types."""
    result = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, list):
            result[k] = [str(x) for x in v]
        else:
            result[k] = str(v)
    return result
