"""Noise audio perturbation: unrelated audio swap.

Replaces each sample's audio with a different sample's audio from the same
benchmark, simulating a worst-case audio-input corruption while preserving
realistic acoustic properties.
"""

import random
from copy import deepcopy

import torch

from src.core.registry import register_perturbation
from src.core.types import Sample
from src.data.audio_utils import get_audio_sample_rate, audio_to_tempfile
from src.perturbations.base import Perturbation


@register_perturbation("noise_audio")
class NoiseAudio(Perturbation):
    """Replace audio with an unrelated sample's audio from the same benchmark."""

    def __init__(self):
        self._benchmark = None  # lazy benchmark reference (set by init_context)
        self._pool_size = 0

    @property
    def name(self) -> str:
        return "noise_audio"

    def init_context(self, benchmark) -> None:
        """Store benchmark reference for lazy audio access.

        Instead of eagerly building an audio pool (which would load every
        sample's audio tensor into RAM), we store the benchmark and access
        audio on-demand via benchmark[idx].audio.
        """
        self._benchmark = benchmark
        self._pool_size = len(benchmark)

    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        s = deepcopy(sample)

        if s.audio is None:
            return s

        return self._apply_unrelated_audio(s, rng)

    def _apply_unrelated_audio(self, sample: Sample, rng: random.Random) -> Sample:
        if self._benchmark is None or self._pool_size <= 1:
            # Not enough samples to swap — return unchanged
            sample.metadata["noise_type"] = "unrelated"
            sample.metadata["noise_swap_failed"] = True
            return sample

        # Try up to 10 times to find a different sample with valid audio
        replacement = None
        for _ in range(10):
            idx = rng.randint(0, self._pool_size - 1)
            other = self._benchmark[idx]
            if other.id != sample.id and other.audio is not None:
                replacement = other.audio
                break

        if replacement is None:
            sample.metadata["noise_type"] = "unrelated"
            sample.metadata["noise_swap_failed"] = True
            return sample

        # Copy replacement audio into the sample
        if isinstance(sample.audio, str) and isinstance(replacement, str):
            sample.audio = replacement
            sample.metadata["perturbation_audio_path"] = replacement
        elif isinstance(sample.audio, torch.Tensor) and isinstance(replacement, torch.Tensor):
            sample.audio = replacement.clone()
        elif isinstance(replacement, str):
            from src.data.audio_utils import load_audio
            replacement_tensor, _ = load_audio(replacement)
            sample.audio = replacement_tensor
        else:
            from src.data.audio_utils import audio_to_tempfile
            sr = get_audio_sample_rate(sample.audio) if isinstance(sample.audio, str) else 16000
            sample.audio = audio_to_tempfile(replacement, sr)
            sample.metadata["perturbation_audio_path"] = sample.audio

        sample.metadata["noise_type"] = "unrelated"
        return sample
