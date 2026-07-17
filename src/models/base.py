"""Abstract model interface for audio QA inference."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class AbstractModel(ABC):
    """Interface that every model adapter must implement.

    Each model may have different processor requirements, prompt formats,
    and audio handling. The adapter encapsulates these differences.
    """

    @abstractmethod
    def load(self) -> None:
        """Load model weights and processor into memory (onto GPU)."""
        ...

    @abstractmethod
    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        """Run inference on a single sample.

        Args:
            audio: str (file path), torch.Tensor (waveform), or None (text-only).
            question: The question text.
            choices: List of multiple-choice option strings.
            label_only: If True, prompt for label-only output (letter A/B/C/D).

        Returns:
            (chosen_answer, raw_model_output) tuple.
            chosen_answer is extracted via clean_answer(); raw_output is the full decoded text.
        """
        ...

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """Run inference on a batch of samples (default: sequential fallback).

        Override this in model adapters that can batch multiple samples
        into a single ``model.generate()`` call for higher throughput.

        Args:
            batch: List of tuples, each:
                (audio, question, choices, label_only)

        Returns:
            List of (chosen_answer, raw_model_output) tuples, one per sample,
            in the same order as the input batch.
        """
        results = []
        for audio, question, choices, label_only in batch:
            try:
                results.append(self.infer(audio, question, choices, label_only=label_only))
            except Exception:
                results.append((choices[0] if choices else "", "[BATCH_ERROR]"))
        return results

    @abstractmethod
    def unload(self) -> None:
        """Free GPU memory and release model resources."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Model identifier string (e.g. 'qwen_omni')."""
        ...

    @property
    @abstractmethod
    def supports_text_only(self) -> bool:
        """Can this model perform inference without audio input?"""
        ...

    def _tensor_to_tempfile(self, audio: torch.Tensor, sr: int = 16000) -> str:
        """Convert a tensor to a temp WAV file. Caller cleans up."""
        from src.data.audio_utils import audio_to_tempfile
        return audio_to_tempfile(audio, sr)


class ModelLoadError(Exception):
    """Raised when model loading fails."""
    pass


class ModelInferenceError(Exception):
    """Raised when inference fails for a sample."""
    pass
