"""Shared dataclasses, enums, and type aliases for the evaluation framework."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Union, Optional

import torch


# ---------------------------------------------------------------------------
# Core data units
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """Universal data unit flowing through the pipeline.

    audio can be:
      - str: path to an audio file
      - torch.Tensor: raw audio waveform
      - None: text-only mode
    """
    id: str
    audio: Union[str, torch.Tensor, None]
    question: str
    choices: list[str]
    ground_truth: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A single model prediction after inference and answer extraction."""
    sample_id: str
    question: str
    choices: list[str]
    ground_truth: str
    chosen_answer: str
    raw_output: str
    correct: bool
    strict_correct: bool = False
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    perturbation: str = ""
    model: str = ""
    benchmark: str = ""


@dataclass
class ResultSet:
    """Container for all predictions from one evaluation task."""
    model: str
    benchmark: str
    perturbation: str
    predictions: list[Prediction] = field(default_factory=list)
    errors: int = 0
    total: int = 0
    # Timing fields (seconds)
    duration_seconds: float = 0.0
    samples_per_second: float = 0.0
    perturbation_time_seconds: float = 0.0
    inference_time_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return sum(1 for p in self.predictions if p.correct) / self.total

    @property
    def strict_accuracy(self) -> float:
        if self.total == 0:
            return 0.0
        return sum(1 for p in self.predictions if p.strict_correct) / self.total

    @property
    def correct_count(self) -> int:
        return sum(1 for p in self.predictions if p.correct)

    @property
    def strict_correct_count(self) -> int:
        return sum(1 for p in self.predictions if p.strict_correct)


# ---------------------------------------------------------------------------
# Configuration models (Pydantic for validation)
# ---------------------------------------------------------------------------

from pydantic import BaseModel as _BaseModel, Field as _Field

class ModelConfig(_BaseModel):
    name: str
    display_name: str = ""
    hf_model_id: str = ""
    local_path: str = ""
    dtype: str = "bfloat16"
    device_map: str = "auto"
    max_memory: Optional[dict] = None
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    supports_text_only: bool = True
    processor_kwargs: dict = _Field(default_factory=dict)
    generate_kwargs: dict = _Field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.local_path or self.hf_model_id


class BenchmarkConfig(_BaseModel):
    name: str
    display_name: str = ""
    hf_dataset: str = ""
    split: str = "test"
    max_samples: Optional[int] = None
    audio_field: str = "audio"
    audio_root: Optional[str] = None
    category_fields: list[str] = _Field(default_factory=list)


class PerturbationConfig(_BaseModel):
    name: str
    params: dict = _Field(default_factory=dict)


class RuntimeConfig(_BaseModel):
    device: str = "cuda:0"
    batch_size: int = 1
    max_new_tokens: int = 64
    do_sample: bool = False
    num_beams: int = 1
    save_predictions: bool = True
    resume: bool = True
    log_level: str = "INFO"


class ExperimentConfig(_BaseModel):
    name: str
    description: str = ""
    seed: int = 42
    output_dir: str = "outputs"
    models: list[str] = _Field(default_factory=list)
    benchmarks: list[str] = _Field(default_factory=list)
    perturbations: list[str] = _Field(default_factory=list)
    runtime: RuntimeConfig = _Field(default_factory=RuntimeConfig)
