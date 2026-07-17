"""Baseline perturbation: no transformation."""

import random
from copy import deepcopy

from src.core.registry import register_perturbation
from src.core.types import Sample
from src.perturbations.base import Perturbation


@register_perturbation("baseline")
class Baseline(Perturbation):
    """Identity perturbation. Returns the sample unchanged."""

    @property
    def name(self) -> str:
        return "baseline"

    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        return deepcopy(sample)
