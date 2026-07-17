"""Silent audio perturbation: replace audio with silence (all zeros)."""

import os
import random
from copy import deepcopy

import torch

from src.core.registry import register_perturbation
from src.core.types import Sample
from src.data.audio_utils import get_audio_duration, get_audio_sample_rate, generate_silence, audio_to_tempfile
from src.perturbations.base import Perturbation


@register_perturbation("silent_audio")
class SilentAudio(Perturbation):
    """Replace audio waveform with silence of the same duration.

    For path-based audio: generates a silent WAV file.
    For tensor-based audio: creates a zero tensor of same shape.
    """

    @property
    def name(self) -> str:
        return "silent_audio"

    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        s = deepcopy(sample)

        if s.audio is None:
            return s

        duration = get_audio_duration(s.audio)
        if duration <= 0:
            return s

        if isinstance(s.audio, str):
            # Path-based: generate silent WAV
            sr = get_audio_sample_rate(s.audio)
            silent = generate_silence(duration, sr)
            s.audio = audio_to_tempfile(silent, sr)
            s.metadata["perturbation_audio_path"] = s.audio
        elif isinstance(s.audio, torch.Tensor):
            # Tensor-based: zero out
            s.audio = torch.zeros_like(s.audio)

        s.metadata["original_audio_duration"] = duration
        return s
