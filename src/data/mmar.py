"""MMAR benchmark loader.

MMAR (Massive Multi-disciplinary Audio Reasoning) from NeurIPS 2025.
HuggingFace: BoJack/MMAR
- 1,000 questions with public answers
- Audio: file path strings (resolved against audio_root)
- 7 modality types, 4 reasoning layers

Supports three loading modes:
  1. HF dataset name → downloads from HuggingFace
  2. Local JSONL/JSON file path → reads from disk (offline)
  3. HF_DATASETS_OFFLINE=1 → uses cached HF datasets

Design note (memory): MMAR audio is file-path strings (not embedded WAV),
so AudioDecoder is not a concern.  However we still drop all HF dataset
references after load so the Arrow table can be reclaimed — matching the
pattern in mmau.py and legacy/infer.py (audio = plain string, no lazy HF
access).
"""

import json
import os
from typing import Optional

from src.core.types import Sample
from src.core.registry import register_benchmark
from src.data.base import AbstractBenchmark


@register_benchmark("mmar")
class MMARBenchmark(AbstractBenchmark):
    def __init__(self, hf_dataset: str = "BoJack/MMAR", split: str = "test",
                 max_samples: Optional[int] = None, audio_root: Optional[str] = None):
        self._hf_dataset = hf_dataset
        self._split = split
        self._max_samples = max_samples
        self._audio_root = audio_root or ""
        self._data: list[Sample] = []
        self._loaded = False
        self._source_type: str = 'file'  # 'hf' or 'file'

    @property
    def name(self) -> str:
        return "mmar"

    @property
    def category_fields(self) -> list[str]:
        return ["modality", "category", "sub_category", "language", "source"]

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
    #  HF-dataset loading
    # ------------------------------------------------------------------

    def _load_from_hf(self) -> None:
        """Load metadata + resolve audio paths from an HF dataset.

        MMAR stores audio as file-path strings (not embedded WAV bytes),
        so we only need to resolve paths against audio_root.  We also
        compute audio duration from the WAV header where possible for
        sorted batching.

        We drop the HF dataset reference after loading so the Arrow table
        can be GC'd — never holding a live ``datasets`` reference avoids
        subtle memory issues (see mmau.py for the hard-leak scenario with
        AudioDecoder).
        """
        from datasets import load_dataset as hf_load_dataset

        ds = hf_load_dataset(self._hf_dataset, split=self._split)
        n_total = len(ds)
        n = min(self._max_samples, n_total) if self._max_samples is not None else n_total

        # Strip audio column if present (MMAR stores paths in "audio_path")
        if "audio" in ds.column_names:
            meta_ds = ds.remove_columns(["audio"])
        else:
            meta_ds = ds

        self._source_type = 'hf'

        # Resolve ALL paths eagerly so we can drop the HF ref immediately
        import soundfile as sf
        samples: list[Sample] = []
        for i in range(n):
            item = meta_ds[i]
            sample = self._parse_one(item, i)

            # Compute audio duration from WAV header where possible
            duration = 0.0
            if sample.audio is not None and isinstance(sample.audio, str) and os.path.exists(sample.audio):
                try:
                    duration = sf.info(sample.audio).duration
                except Exception:
                    duration = 0.0
            sample.metadata["audio_duration"] = duration

            samples.append(sample)

        # Drop ALL HF dataset references — Arrow table can now be GC'd
        del ds, meta_ds

        self._data = samples
        print(f"Loaded {len(self._data)} samples from HF dataset: {self._hf_dataset}")

    # ------------------------------------------------------------------
    #  Parsing helpers
    # ------------------------------------------------------------------

    def _parse_items(self, items: list[dict]) -> list[Sample]:
        """Parse samples from a list of dicts."""
        indices = range(len(items))
        if self._max_samples is not None:
            indices = range(min(self._max_samples, len(items)))
        return [self._parse_one(items[i], i) for i in indices]

    def _parse_one(self, item: dict, idx: int) -> Sample:
        """Parse a single sample from a dict (HF item or JSON dict)."""
        # Resolve audio path — may be a plain string or a dict with "path" key
        audio_path = item.get("audio_path", item.get("audio", ""))
        if isinstance(audio_path, dict):
            audio_path = audio_path.get("path", audio_path.get("filename", ""))
        if audio_path and self._audio_root and not os.path.isabs(str(audio_path)):
            audio_path = os.path.join(self._audio_root, str(audio_path))

        return Sample(
            id=str(item.get("id", f"mmar_{idx}")),
            audio=str(audio_path) if audio_path else None,
            question=item.get("question", ""),
            choices=item.get("choices", []),
            ground_truth=item.get("answer", ""),
            metadata={
                "modality": item.get("modality", ""),
                "category": item.get("category", ""),
                "sub_category": item.get("sub_category", item.get("sub-category", "")),
                "language": item.get("language", ""),
                "source": item.get("source", ""),
            },
        )

    # ------------------------------------------------------------------
    #  AbstractBenchmark interface
    # ------------------------------------------------------------------

    @property
    def audio_samples(self) -> list:
        """Return audio path references for unrelated-audio perturbation."""
        return [s.audio for s in self._data]

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Sample:
        if not self._loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")
        return self._data[idx]
