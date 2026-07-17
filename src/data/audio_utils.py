"""Audio utility functions: loading, resampling, generation."""

import os
import tempfile
from typing import Optional, Tuple

import numpy as np
import torch
import soundfile as sf

DEFAULT_SAMPLE_RATE = 16000


def load_audio(path: str, target_sr: int = DEFAULT_SAMPLE_RATE) -> Tuple[torch.Tensor, int]:
    """Load audio from file and resample to target sample rate.

    Returns (waveform_tensor, sample_rate).
    """
    import librosa
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    return torch.from_numpy(audio).float(), sr


def get_audio_duration(audio) -> float:
    """Get duration of audio in seconds.

    audio can be:
      - str: path to audio file
      - torch.Tensor: raw waveform (assumes 16kHz sample rate)
      - np.ndarray: raw waveform
    """
    if isinstance(audio, str):
        return sf.info(audio).duration
    elif isinstance(audio, torch.Tensor):
        return audio.numel() / DEFAULT_SAMPLE_RATE
    elif isinstance(audio, np.ndarray):
        return audio.size / DEFAULT_SAMPLE_RATE
    return 0.0


def get_audio_sample_rate(audio) -> int:
    """Get sample rate of audio.

    audio can be str path or torch.Tensor.
    For tensors, assumes DEFAULT_SAMPLE_RATE.
    """
    if isinstance(audio, str):
        try:
            return sf.info(audio).samplerate
        except Exception:
            return DEFAULT_SAMPLE_RATE
    return DEFAULT_SAMPLE_RATE


def generate_silence(duration_s: float, sr: int = DEFAULT_SAMPLE_RATE) -> torch.Tensor:
    """Generate a silent audio tensor (all zeros)."""
    num_samples = int(duration_s * sr)
    return torch.zeros(num_samples, dtype=torch.float32)


def generate_white_noise(duration_s: float, sr: int = DEFAULT_SAMPLE_RATE,
                         rng: Optional[np.random.RandomState] = None) -> torch.Tensor:
    """Generate white noise audio tensor."""
    if rng is None:
        rng = np.random.RandomState()
    num_samples = int(duration_s * sr)
    noise = rng.randn(num_samples).astype(np.float32)
    # Normalize to avoid clipping
    noise = noise / (np.abs(noise).max() + 1e-8) * 0.95
    return torch.from_numpy(noise)


def audio_to_tempfile(audio: torch.Tensor, sr: int = DEFAULT_SAMPLE_RATE,
                      format: str = "wav") -> str:
    """Write an audio tensor to a temporary WAV file. Returns the file path.

    Caller is responsible for cleaning up the temp file.
    """
    fd, path = tempfile.mkstemp(suffix=f".{format}")
    os.close(fd)
    waveform = audio.cpu().numpy()
    sf.write(path, waveform, sr)
    return path


def get_audio_tensor(audio) -> Optional[torch.Tensor]:
    """Get audio as a torch.Tensor regardless of input type.

    Returns None if audio is None.
    """
    if audio is None:
        return None
    if isinstance(audio, torch.Tensor):
        return audio
    if isinstance(audio, str):
        tensor, _ = load_audio(audio)
        return tensor
    if isinstance(audio, np.ndarray):
        return torch.from_numpy(audio).float()
    return None
