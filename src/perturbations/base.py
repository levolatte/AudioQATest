"""Abstract perturbation interface."""

import random
from abc import ABC, abstractmethod
from typing import Any

from src.core.types import Sample


class Perturbation(ABC):
    """Interface for all perturbation conditions.

    Each perturbation applies a transformation to a Sample and returns
    a NEW Sample object (the original is never mutated).

    Args:
        sample: The original sample to transform.
        rng: A random.Random instance for reproducible randomness.
        **kwargs: Perturbation-specific parameters.
    """

    @abstractmethod
    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        """Transform a sample and return a new Sample."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Perturbation identifier."""
        ...

    def init_context(self, benchmark: "AbstractBenchmark") -> None:
        """Called once before any apply() calls.

        Args:
            benchmark: The benchmark instance. Subclasses can store a
                       reference and access audio samples lazily via
                       benchmark[idx].audio.
        """
        pass
