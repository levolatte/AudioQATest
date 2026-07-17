"""CombinedEvaluationTask: batch multiple perturbations of the same samples.

Key insight: when different perturbations are applied to the **same** sample,
the audio is identical (baseline, shuffled, label_only) or nearly identical
(silent = same duration, noise = maybe different).  Grouping them into one
batch reduces padding waste from O(N) to O(N/P) where P is the number of
perturbations.

Example — 3 samples × 5 perturbations = 15 items per batch but only ~3
unique audio lengths, vs 12 items with 12 different lengths in the old
single-perturbation batching.
"""

import json
import os
import random
import time
from typing import Optional, Set

import soundfile as sf
import torch
from tqdm import tqdm

from src.core.types import Sample, Prediction, ResultSet, PerturbationConfig
from src.data.base import AbstractBenchmark
from src.models.base import AbstractModel
from src.perturbations.base import Perturbation
from src.runner.logging_setup import get_task_logger


# Re-use the prediction serialisation helper from task.py
def _pred_to_dict(p: Prediction) -> dict:
    d = {
        "sample_id": p.sample_id,
        "question": p.question,
        "choices": p.choices,
        "ground_truth": p.ground_truth,
        "chosen_answer": p.chosen_answer,
        "raw_output": p.raw_output,
        "correct": p.correct,
        "error": p.error,
        "metadata": _serialize_metadata(p.metadata),
        "perturbation": p.perturbation,
        "model": p.model,
        "benchmark": p.benchmark,
    }
    return d


def _serialize_metadata(meta: dict) -> dict:
    result = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool, type(None))):
            result[k] = v
        elif isinstance(v, list):
            result[k] = [str(x) for x in v]
        else:
            result[k] = str(v)
    return result


class CombinedEvaluationTask:
    """Runs one model + one benchmark with ALL perturbations batched together.

    Instead of running each perturbation as a separate task (which means
    each batch contains different audio files of varying lengths), this
    class applies all perturbations to each sample and groups the results
    into a single batch — minimising padding waste inside the audio encoder.
    """

    def __init__(
        self,
        model: AbstractModel,
        benchmark: AbstractBenchmark,
        perturbations: list[tuple[str, Perturbation, PerturbationConfig]],
        output_dir: str,
        samples_per_batch: int = 3,
        seed: int = 42,
        skip_completed: bool = True,
    ):
        """
        Args:
            perturbations: List of (perturbation_id, instance, config) tuples.
            samples_per_batch: Number of **unique samples** per batch.
                Total batch size = samples_per_batch × len(perturbations).
        """
        self.model = model
        self.benchmark = benchmark
        self.perturbations = perturbations  # [(id, instance, config), ...]
        self.samples_per_batch = samples_per_batch
        self.total_batch_size = samples_per_batch * len(perturbations)
        self.output_dir = output_dir
        self.seed = seed
        self._rng = random.Random(seed)
        self._skip_completed = skip_completed

    @property
    def task_id(self) -> str:
        return f"{self.model.name}__{self.benchmark.name}__combined"

    def run(self) -> dict[str, ResultSet]:
        """Execute all perturbation tasks with combined batching.

        Returns:
            Dict mapping perturbation_id -> ResultSet.
        """
        pred_dir = os.path.join(self.output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # ── Per-perturbation prediction files ──
        pred_files: dict[str, str] = {}
        for pname, _, _ in self.perturbations:
            tid = f"{self.model.name}__{self.benchmark.name}__{pname}"
            pred_files[pname] = os.path.join(pred_dir, f"{tid}.jsonl")

        logger = get_task_logger(
            model=self.model.name,
            benchmark=self.benchmark.name,
            perturbation="combined",
            log_dir=log_dir,
        )

        logger.info(f"=== Combined task start: {self.task_id} ===")
        logger.info(f"Model: {self.model.name}, Benchmark: {self.benchmark.name}, "
                    f"Perturbations: {[p[0] for p in self.perturbations]}, "
                    f"Samples: {len(self.benchmark)}, "
                    f"Samples per batch: {self.samples_per_batch} "
                    f"(total batch size: {self.total_batch_size})")

        # ── Load existing predictions (resume per perturbation) ──
        completed: dict[str, set] = {pname: set() for pname, _, _ in self.perturbations}
        predictions: dict[str, list[Prediction]] = {pname: [] for pname, _, _ in self.perturbations}

        if self._skip_completed:
            for pname, pfile in pred_files.items():
                if os.path.exists(pfile):
                    with open(pfile, "r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                pred = json.loads(line.strip())
                                completed[pname].add(pred["sample_id"])
                                predictions[pname].append(Prediction(**pred))
                            except (json.JSONDecodeError, TypeError):
                                continue
            total_done = sum(len(c) for c in completed.values())
            total_possible = len(self.benchmark) * len(self.perturbations)
            logger.info(f"Resume: {total_done}/{total_possible} predictions already completed")

        # ── Pre-compute audio durations for sorted batching ──
        # Prefer metadata "audio_duration" (set by lazy-loading benchmarks
        # from the WAV header without decoding) so that sorting doesn't
        # force eager audio extraction.
        sample_order: list[tuple[int, float]] = []
        for i in range(len(self.benchmark)):
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
            logger.info(f"Sorted {len(sample_order)} samples by audio duration "
                        f"(range: {sample_order[0][1]:.1f}s – {sample_order[-1][1]:.1f}s)")

        # ── Timing ──
        errors = 0
        start_time = time.time()
        pert_time_total = 0.0
        infer_time_total = 0.0

        pbar = tqdm(total=len(sample_order), desc=self.task_id, unit="samples")

        # ── Main loop: accumulate perturbations-per-sample into batches ──
        # pending: list of (perturbation_id, original_sample_idx, transformed_sample)
        pending: list[tuple[str, int, Sample]] = []

        for idx, _dur in sample_order:
            sample = self.benchmark[idx]

            for pname, perturbation, pconfig in self.perturbations:
                if sample.id in completed.get(pname, set()):
                    continue

                pert_start = time.time()
                try:
                    transformed = perturbation.apply(sample, self._rng, **pconfig.params)
                except Exception as e:
                    logger.error(f"Perturbation {pname} failed for {sample.id}: {e}")
                    errors += 1
                    continue
                pert_time_total += time.time() - pert_start

                pending.append((pname, idx, transformed))

            # Flush when we have enough items
            while len(pending) >= self.total_batch_size:
                batch = pending[:self.total_batch_size]
                infer_time_total += self._flush_batch(
                    batch, predictions, pred_files, completed, logger,
                )
                pending = pending[self.total_batch_size:]
                pbar.update(self.samples_per_batch)

        # Flush remainder
        if pending:
            infer_time_total += self._flush_batch(
                pending, predictions, pred_files, completed, logger,
            )
            # Approximate: the remainder corresponds to roughly
            # ceil(len(remainder)/num_perturbs) unique samples
            remainder_samples = (len(pending) + len(self.perturbations) - 1) // len(self.perturbations)
            pbar.update(remainder_samples)

        pbar.close()

        # ── Build per-perturbation ResultSets ──
        elapsed = time.time() - start_time
        result_sets: dict[str, ResultSet] = {}

        for pname, _, _ in self.perturbations:
            preds = predictions[pname]
            n_correct = sum(1 for p in preds if p.correct)
            rs = ResultSet(
                model=self.model.name,
                benchmark=self.benchmark.name,
                perturbation=pname,
                predictions=preds,
                errors=errors,
                total=len(self.benchmark),
                duration_seconds=round(elapsed, 3),
                samples_per_second=round(len(preds) / elapsed, 3) if elapsed > 0 else 0,
                perturbation_time_seconds=round(pert_time_total, 3),
                inference_time_seconds=round(infer_time_total, 3),
            )
            result_sets[pname] = rs
            logger.info(f"  [{pname}] accuracy={rs.accuracy:.4f} ({n_correct}/{len(preds)})")

        logger.info(f"=== Combined task complete: {self.task_id} ===")
        logger.info(f"Duration: {elapsed:.1f}s | "
                    f"Throughput: {len(self.benchmark) * len(self.perturbations) / elapsed:.2f} items/s")

        return result_sets

    def _flush_batch(
        self,
        pending: list[tuple[str, int, Sample]],
        predictions: dict[str, list[Prediction]],
        pred_files: dict[str, str],
        completed: dict[str, set],
        logger,
    ) -> float:
        """Run batch inference on a mixed-perturbation batch and write results.

        Returns elapsed inference time in seconds.
        """
        batch_inputs: list[tuple] = []
        for _, _, sample in pending:
            batch_inputs.append((
                sample.audio,
                sample.question,
                sample.choices,
                sample.metadata.get("label_only", False),
            ))

        infer_start = time.time()
        try:
            batch_results = self.model.infer_batch(batch_inputs)
        except Exception as e:
            logger.error(f"Batch inference failed ({len(pending)} items): {e}")
            torch.cuda.empty_cache()
            # Fall back to sequential
            batch_results = []
            for audio, question, choices, label_only in batch_inputs:
                try:
                    chosen, raw = self.model.infer(audio, question, choices,
                                                   label_only=label_only)
                    batch_results.append((chosen, raw))
                except Exception as e2:
                    logger.error(f"Sequential inference failed: {e2}")
                    batch_results.append((choices[0] if choices else "", f"[ERROR] {e2}"))
        infer_elapsed = time.time() - infer_start

        for j, (pname, _, sample) in enumerate(pending):
            chosen, raw = batch_results[j]
            correct = (chosen == sample.ground_truth)

            prediction = Prediction(
                sample_id=sample.id,
                question=sample.question,
                choices=sample.choices,
                ground_truth=sample.ground_truth,
                chosen_answer=chosen,
                raw_output=raw,
                correct=correct,
                error=None,
                metadata=sample.metadata,
                perturbation=pname,
                model=self.model.name,
                benchmark=self.benchmark.name,
            )

            predictions[pname].append(prediction)

            with open(pred_files[pname], "a", encoding="utf-8") as f:
                f.write(json.dumps(_pred_to_dict(prediction), ensure_ascii=False) + "\n")

            completed[pname].add(sample.id)

        return infer_elapsed
