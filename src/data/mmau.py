"""MMAU benchmark loader.

MMAU (Massive Multi-Task Audio Understanding) from ICLR 2025.
HuggingFace: lmms-lab/mmau
- test-mini: 1,000 questions with public answers
- test: 9,000 questions (answers withheld)
- Audio: embedded WAV bytes via HF Audio feature

Supports three loading modes:
  1. HF dataset name → downloads from HuggingFace
  2. Local JSONL/JSON file path → reads from disk (offline)
  3. HF_DATASETS_OFFLINE=1 → uses cached HF datasets

Design note (memory): HF datasets' AudioDecoder (torchcodec) has a C++
level memory leak — decoded audio is cached and never released by Python
GC.  To avoid crashes on 1000-sample runs, we bypass AudioDecoder
entirely: raw WAV bytes are read directly from the Arrow table column
and decoded with soundfile.

Lazy-loading strategy (since 2026-07):
  - ``load()`` stores only metadata + duration (parsed from WAV header);
    audio is NOT extracted yet.  The Arrow column handle is kept alive.
  - ``__getitem__()`` extracts audio to a temp WAV on first access and
    caches the path in the Sample.  This spreads extraction over time
    instead of peaking during load().
"""

import atexit
import io
import json
import os
import shutil
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from src.core.types import Sample
from src.core.registry import register_benchmark
from src.data.base import AbstractBenchmark


@register_benchmark("mmau")
class MMAUBenchmark(AbstractBenchmark):
    def __init__(self, hf_dataset: str = "lmms-lab/mmau", split: str = "test_mini",
                 max_samples: Optional[int] = None, audio_root: Optional[str] = None):
        self._hf_dataset = hf_dataset
        self._split = split
        self._max_samples = max_samples
        self._audio_root = audio_root  # accepted but unused (MMAU audio is embedded)
        self._data: list[Sample] = []
        self._loaded = False
        self._temp_dir = None
        self._source_type: str = 'file'       # 'hf' or 'file'
        self._hf_audio_arrow = None           # Arrow ChunkedArray for lazy audio
        atexit.register(self._cleanup_temp_files)

    @property
    def name(self) -> str:
        return "mmau"

    @property
    def category_fields(self) -> list[str]:
        return ["dataset", "task", "category", "sub_category", "difficulty"]

    def load(self, split: Optional[str] = None) -> None:
        if split:
            self._split = split

        # Mode 1: Local file path
        if os.path.isfile(self._hf_dataset):
            self._load_from_file(self._hf_dataset)
        else:
            # Mode 2/3: HF dataset (online or offline cache)
            self._load_from_hf()

        self._loaded = True

    # ------------------------------------------------------------------
    #  File-based loading (local JSON / JSONL)
    # ------------------------------------------------------------------

    def _load_from_file(self, path: str) -> None:
        """Load from a local JSON or JSONL file."""
        if path.endswith(".jsonl"):
            with open(path, "r", encoding="utf-8") as f:
                items = [json.loads(line) for line in f if line.strip()]
        elif path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else data.get("data", [])
        else:
            raise ValueError(f"Unsupported file format: {path}. Expected .json or .jsonl")

        self._source_type = 'file'
        self._data = self._parse_items(items)
        print(f"Loaded {len(self._data)} samples from local file: {path}")

    # ------------------------------------------------------------------
    #  HF-dataset loading  (the tricky part — must avoid AudioDecoder)
    # ------------------------------------------------------------------

    def _load_from_hf(self) -> None:
        """Load **metadata only** from an HF dataset; audio is extracted lazily.

        Audio is stored as raw WAV bytes in an Arrow ``struct<bytes, path>``
        column.  We keep the Arrow column handle alive and compute each
        sample's duration from the WAV header (cheap — no decoding).

        Actual audio extraction → temp WAV happens on the first
        ``__getitem__`` call, spreading the cost over the experiment
        instead of peaking during ``load()``.
        """
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset(self._hf_dataset, split=self._split)
        n_total = len(ds)
        n = min(self._max_samples, n_total) if self._max_samples is not None else n_total

        # --- audio: get raw Arrow column handle (keep alive for lazy reads) ---
        if "audio" in ds.column_names:
            self._hf_audio_arrow = ds.data.column("audio")
        else:
            self._hf_audio_arrow = None

        # --- metadata: strip audio column so iteration is cheap ---
        meta_ds = ds.remove_columns(["audio"]) if "audio" in ds.column_names else ds

        # --- temp directory for extracted WAV files ---
        self._temp_dir = tempfile.mkdtemp(prefix="mmau_audio_")
        self._source_type = 'hf'

        samples: list[Sample] = []
        for i in range(n):
            item = meta_ds[i]

            # -- compute duration from WAV header (fast, no decoding) --------
            duration = 0.0
            if self._hf_audio_arrow is not None:
                try:
                    cell = self._hf_audio_arrow[i].as_py()
                    raw_bytes = cell.get("bytes", None) if isinstance(cell, dict) else None
                    if raw_bytes:
                        duration = sf.info(io.BytesIO(raw_bytes)).duration
                except Exception:
                    duration = 0.0

            # -- parse choices (HF stores them as JSON strings) ---------------
            raw_choices = item.get("choices", [])
            if isinstance(raw_choices, str):
                try:
                    raw_choices = json.loads(raw_choices)
                except (json.JSONDecodeError, TypeError):
                    raw_choices = []
            if not isinstance(raw_choices, list):
                raw_choices = []

            sample = Sample(
                id=str(item.get("id", f"mmau_{i}")),
                audio=None,  # ← lazy: extracted on first __getitem__ access
                question=item.get("question", ""),
                choices=raw_choices,
                ground_truth=item.get("answer", ""),
                metadata={
                    "dataset":      item.get("dataset", ""),
                    "task":         item.get("task", ""),
                    "category":     item.get("category", ""),
                    "sub_category": item.get("sub-category",
                                             item.get("sub_category", "")),
                    "difficulty":   item.get("difficulty", ""),
                    "split":        item.get("split", ""),
                    "audio_duration": duration,   # for sorted batching
                },
            )
            samples.append(sample)

        # Drop the metadata-side references (keep _hf_audio_arrow alive).
        del ds, meta_ds

        self._data = samples
        print(f"Loaded {len(self._data)} samples (metadata only) from HF dataset: {self._hf_dataset}")

    def _extract_audio_sample(self, audio_arrow, i: int) -> Optional[str]:
        """Decode the i-th audio sample from the Arrow column → 16 kHz WAV.

        Uses raw bytes + soundfile (NO AudioDecoder).  Returns the path
        to the temp WAV file, or None if the sample has no audio.
        """
        try:
            cell = audio_arrow[i].as_py()
        except Exception:
            return None

        raw_bytes = cell.get("bytes", None) if isinstance(cell, dict) else None
        if raw_bytes is None:
            return None

        arr, sr = sf.read(io.BytesIO(raw_bytes))
        if arr.ndim > 1:
            arr = arr.mean(axis=1)

        # Resample to 16 kHz if necessary
        if sr != 16000:
            t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
            import torchaudio
            t = torchaudio.functional.resample(t, orig_freq=sr, new_freq=16000).squeeze(0)
            arr = t.numpy()
            del t

        path = os.path.join(self._temp_dir, f"sample_{i:04d}.wav")
        sf.write(path, arr.astype(np.float32), 16000)

        # Help GC — explicitly release numpy array
        del arr, raw_bytes, cell
        return path

    # ------------------------------------------------------------------
    #  Cleanup
    # ------------------------------------------------------------------

    def _cleanup_temp_files(self):
        """Remove the temp directory and all extracted audio WAV files."""
        if self._temp_dir is not None and os.path.isdir(self._temp_dir):
            try:
                shutil.rmtree(self._temp_dir)
                self._temp_dir = None
            except OSError:
                pass

    # ------------------------------------------------------------------
    #  Legacy parsing helpers (used by _load_from_file path)
    # ------------------------------------------------------------------

    def _parse_hf_dataset(self, ds) -> list[Sample]:
        """Parse samples from a HuggingFace dataset object."""
        indices = range(len(ds))
        if self._max_samples is not None:
            indices = range(min(self._max_samples, len(ds)))
        return [self._parse_one(ds[i], i) for i in indices]

    def _parse_items(self, items: list[dict]) -> list[Sample]:
        """Parse samples from a list of dicts."""
        indices = range(len(items))
        if self._max_samples is not None:
            indices = range(min(self._max_samples, len(items)))
        return [self._parse_one(items[i], i) for i in indices]

    def _resolve_hf_audio(self, audio_val):
        """Convert an HF Audio feature to torch.Tensor or str path.

        Handles:
          - dict with 'array' (numpy) and 'sampling_rate'
          - dict with 'path' (str or AudioDecoder object)
          - torchcodec AudioDecoder (directly or inside dict)
          - str (file path) — kept as-is
          - torch.Tensor — kept as-is
          - None — returned as-is

        ALWAYS resamples to 16 kHz to keep memory usage bounded.
        """
        if audio_val is None:
            return None

        if isinstance(audio_val, torch.Tensor):
            return audio_val
        if isinstance(audio_val, str):
            if self._audio_root and not os.path.isabs(audio_val):
                return os.path.join(self._audio_root, audio_val)
            return audio_val

        # dict (HF Audio feature)
        if isinstance(audio_val, dict):
            arr = audio_val.get("array", audio_val.get("path", None))
            orig_sr = audio_val.get("sampling_rate", None)

            if isinstance(arr, str):
                if self._audio_root and not os.path.isabs(arr):
                    return os.path.join(self._audio_root, arr)
                return arr

            if hasattr(arr, "shape"):                          # numpy array
                t = torch.from_numpy(arr).float()
                del arr                                 # release numpy allocation
                if t.ndim > 1:
                    t = t.mean(dim=0)
                if orig_sr is not None and orig_sr != 16000:
                    import torchaudio as _ta
                    t = _ta.functional.resample(t, orig_freq=orig_sr, new_freq=16000)
                return t

            if hasattr(arr, 'get_all_samples'):                # AudioDecoder in dict
                return self._decode_audiodecoder(arr, orig_sr)

            return None

        # AudioDecoder at top level
        if hasattr(audio_val, 'get_all_samples'):
            return self._decode_audiodecoder(audio_val, None)

        return None

    def _decode_audiodecoder(self, decoder, orig_sr: Optional[int] = None) -> torch.Tensor:
        """Decode a torchcodec AudioDecoder to a 16 kHz mono tensor.

        .. warning::
           This method is ONLY used by the ``_load_from_file`` path for
           edge-case local JSON files that embed AudioDecoder objects.
           The primary ``_load_from_hf`` path **never** calls this and
           instead uses Arrow raw bytes + soundfile to avoid the C++-level
           memory leak in torchcodec.
        """
        import torchaudio as _ta
        samples = decoder.get_all_samples()
        t = samples.data.float()
        del samples                        # release torchcodec allocation
        if t.ndim > 1:
            t = t.mean(dim=0)
        dur = getattr(decoder.metadata, 'duration_seconds', None) if hasattr(decoder, 'metadata') else None
        n = t.shape[-1]
        if dur and dur > 0:
            actual_sr = int(round(n / dur))
        else:
            actual_sr = orig_sr or getattr(decoder.metadata, 'sample_rate', 16000) if hasattr(decoder, 'metadata') else 16000
        if actual_sr != 16000:
            t = _ta.functional.resample(t, orig_freq=actual_sr, new_freq=16000)
        return t

    def _parse_one(self, item: dict, idx: int) -> Sample:
        """Parse a single sample from a dict (HF item or JSON dict)."""
        audio = self._resolve_hf_audio(item.get("audio", None))

        if audio is None:
            alt_path = item.get("audio_path", None)
            if alt_path:
                if self._audio_root and not os.path.isabs(alt_path):
                    alt_path = os.path.join(self._audio_root, alt_path)
                audio = alt_path

        raw_choices = item.get("choices", [])
        if isinstance(raw_choices, str):
            try:
                raw_choices = json.loads(raw_choices)
            except (json.JSONDecodeError, TypeError):
                raw_choices = []
        if not isinstance(raw_choices, list):
            raw_choices = []

        return Sample(
            id=str(item.get("id", f"mmau_{idx}")),
            audio=audio,
            question=item.get("question", ""),
            choices=raw_choices,
            ground_truth=item.get("answer", ""),
            metadata={
                "dataset": item.get("dataset", ""),
                "task": item.get("task", ""),
                "category": item.get("category", ""),
                "sub_category": item.get("sub-category", item.get("sub_category", "")),
                "difficulty": item.get("difficulty", ""),
                "split": item.get("split", ""),
            },
        )

    # ------------------------------------------------------------------
    #  AbstractBenchmark interface
    # ------------------------------------------------------------------

    @property
    def audio_samples(self) -> list:
        """Return audio file path references for unrelated-audio perturbation.

        With lazy loading, triggering extraction here would defeat the purpose.
        Instead, callers should use ``benchmark[idx].audio`` which extracts
        on first access.
        """
        return [s.audio for s in self._data]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Sample:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        sample = self._data[idx]

        # ── Lazy audio extraction (HF mode only) ──
        if self._source_type == 'hf' and sample.audio is None and self._hf_audio_arrow is not None:
            audio_path = self._extract_audio_sample(self._hf_audio_arrow, idx)
            if audio_path is not None:
                sample.audio = audio_path  # cache in-place
                # Also update the stored Sample so subsequent accesses are instant
                self._data[idx] = sample

        return sample
