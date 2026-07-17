#!/usr/bin/env python3
"""Batch-size throughput sweep: measure inference speed vs batch_size.

Loads model once, pre-extracts + duration-sorts all audio (one-time cost,
excluded from timing), then sweeps batch sizes.  Also tracks VRAM peak
and padding waste to find the optimal batch_size for production.

Usage:
    conda activate audioqa
    python scripts/batch_sweep.py [--samples 96] [--model qwen_omni]
"""

import argparse
import gc
import sys
import time

sys.path.insert(0, ".")

import torch

# ── trigger registrations (same as orchestrator) ──
import src.data.mmau              # noqa: F401
import src.data.mmar              # noqa: F401
import src.models.qwen_omni       # noqa: F401
import src.models.qwen2_audio     # noqa: F401
import src.models.moss_audio      # noqa: F401
import src.models.kimi_audio      # noqa: F401
import src.models.mock_model      # noqa: F401

from src.core.config import (
    load_model_config, load_benchmark_config,
    CONFIG_DIR, _load_yaml, _apply_hf_settings,
)
from src.core.registry import get_model, get_benchmark

# HF mirror / token from config (before any downloads)
_apply_hf_settings(_load_yaml(CONFIG_DIR / "base.yaml"))


def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def main():
    parser = argparse.ArgumentParser(description="Batch-size throughput sweep")
    parser.add_argument("--model", default="qwen_omni")
    parser.add_argument("--benchmark", default="mmau")
    parser.add_argument("--samples", type=int, default=96,
                        help="Total samples (use a common multiple like 96)")
    parser.add_argument("--batch-sizes", default="1,2,4,8,12,16,24,32",
                        help="Comma-separated batch sizes to test")
    args = parser.parse_args()

    BS_LIST = [int(x.strip()) for x in args.batch_sizes.split(",")]
    BS_LIST = [bs for bs in BS_LIST if bs <= args.samples]
    N = args.samples

    print("=" * 72, flush=True)
    print(f"Batch-size sweep: {args.model}  x  {args.benchmark}", flush=True)
    print(f"Samples: {N}   Batch sizes: {BS_LIST}", flush=True)
    print("=" * 72, flush=True)

    # ── 1. Load model ──
    print("\n[1/4] Loading model ...", flush=True)
    t0 = time.time()
    model_cfg = load_model_config(args.model)
    model_cls = get_model(args.model)
    model_cfg_dict = model_cfg.model_dump()
    model_cfg_dict["path"] = model_cfg.path
    model = model_cls(config=model_cfg_dict)
    model.load()
    print(f"       Loaded in {time.time() - t0:.1f}s", flush=True)

    # ── 2. Load benchmark (metadata only — lazy audio) ──
    print(f"\n[2/4] Loading benchmark ({N} samples, metadata only) ...", flush=True)
    t0 = time.time()
    bench_cfg = load_benchmark_config(args.benchmark)
    bench_cls = get_benchmark(args.benchmark)
    benchmark = bench_cls(
        hf_dataset=bench_cfg.hf_dataset,
        split=bench_cfg.split,
        max_samples=N,
        audio_root=bench_cfg.audio_root,
    )
    benchmark.load()
    print(f"       {len(benchmark)} samples (metadata) in {time.time() - t0:.1f}s", flush=True)

    # ── 3. Extract + sort by duration ──
    #     This triggers lazy audio extraction (one-time cost, excluded from
    #     inference timing).  Sorting by duration matches production batching
    #     and minimises padding waste inside the audio encoder.
    print(f"\n[3/4] Extracting audio + sorting by duration ...", flush=True)
    t0 = time.time()
    samples = []  # (audio, question, choices, label_only, duration)
    for i in range(len(benchmark)):
        s = benchmark[i]          # ← triggers lazy extraction on first access
        dur = s.metadata.get("audio_duration", 0.0) or 0.0
        samples.append((s.audio, s.question, s.choices, False, dur))
    samples.sort(key=lambda x: x[4])
    n_audio = sum(1 for s in samples if s[0] is not None)
    print(f"       {n_audio} audio files extracted + sorted in {time.time() - t0:.1f}s",
          flush=True)

    del benchmark
    gc.collect()

    # ── 4. Sweep ──
    print(f"\n[4/4] Sweeping batch sizes ...\n", flush=True)

    results = []

    for bs in BS_LIST:
        # Build batches
        batches = []
        for chunk_samples in chunk(samples, bs):
            batch = [(a, q, c, lo) for a, q, c, lo, _d in chunk_samples]
            batches.append(batch)
        n_batches = len(batches)

        # Padding waste %
        total_dur = 0.0
        total_waste = 0.0
        for chunk_samples in chunk(samples, bs):
            durs = [s[4] for s in chunk_samples]
            if durs:
                total_dur += sum(durs)
                total_waste += sum(max(durs) - d for d in durs)
        waste_pct = (total_waste / total_dur * 100) if total_dur > 0 else 0.0

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            # Warmup: first batch (not timed)
            _ = model.infer_batch(batches[0])
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # Timed run
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

            t_start = time.perf_counter()
            for batch in batches:
                _ = model.infer_batch(batch)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t_start

            peak_mem = 0
            if torch.cuda.is_available():
                peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)

            sps = N / elapsed if elapsed > 0 else float("inf")
            spb = elapsed / n_batches if n_batches > 0 else 0.0

            results.append((bs, elapsed, sps, spb, peak_mem, waste_pct, n_batches, True))
            print(f"  bs={bs:>3d} | {elapsed:>6.1f}s | {sps:>6.2f} sp/s | "
                  f"batch {spb:>5.1f}s | VRAM {peak_mem:>5.0f}M | "
                  f"waste {waste_pct:>4.1f}% | {n_batches:>3d} batches",
                  flush=True)

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            results.append((bs, None, None, None, None, None, n_batches, False))
            print(f"  bs={bs:>3d} | OOM — stopping sweep", flush=True)
            torch.cuda.empty_cache()
            break
        except Exception as e:
            results.append((bs, None, None, None, None, None, n_batches, False))
            print(f"  bs={bs:>3d} | ERROR: {e}", flush=True)
            torch.cuda.empty_cache()

    # ── 5. Report ──
    print("\n" + "=" * 72, flush=True)
    print("RESULTS", flush=True)
    print("=" * 72, flush=True)

    valid = [r for r in results if r[2] is not None]
    if valid:
        best = max(valid, key=lambda r: r[2])
        baseline = valid[0][2]

        header = (f"\n  {'bs':>5s}  {'time':>8s}  {'sp/s':>8s}  "
                  f"{'s/batch':>8s}  {'VRAM':>7s}  {'waste':>7s}  {'batches':>8s}")
        sep = f"  {'-'*5}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*8}"
        print(header, flush=True)
        print(sep, flush=True)

        for bs, elapsed, sps, spb, peak_mem, waste_pct, n_batches, ok in results:
            if ok:
                flag = "  ★" if bs == best[0] else ""
                print(f"  {bs:>5d}  {elapsed:>7.1f}s  {sps:>7.2f}  "
                      f"{spb:>7.1f}s  {peak_mem:>5.0f}M  {waste_pct:>5.1f}%  "
                      f"{n_batches:>8d}{flag}",
                      flush=True)
            else:
                print(f"  {bs:>5d}  {'FAILED':>8s}", flush=True)

        speedup = best[2] / baseline if baseline > 0 else 0
        print(f"\n  → Best  bs={best[0]}   {best[2]:.2f} samples/s  "
              f"({speedup:.1f}× vs bs=1)", flush=True)

        # Show diminishing returns
        if len(valid) >= 2:
            prev_sps = valid[0][2]
            print("\n  Marginal gain per step:", flush=True)
            for bs, _, sps, _, _, _, _, ok in results:
                if ok and bs > 1:
                    gain = (sps - prev_sps) / prev_sps * 100 if prev_sps > 0 else 0
                    print(f"    bs={bs:>3d}: +{gain:>5.1f}% vs previous", flush=True)
                    prev_sps = sps

    else:
        print("\n  No successful runs!", flush=True)

    # ── Cleanup ──
    print(f"\n[Cleanup] Unloading model ...", flush=True)
    model.unload()
    gc.collect()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
