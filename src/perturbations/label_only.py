"""Label-only perturbation: force label-only output format.

The model outputs ONLY the option letter (A/B/C/D) without any explanation
or option text. Audio is kept unchanged — this is purely an output-format
constraint, not an input perturbation.
"""

import random
from copy import deepcopy

from src.core.registry import register_perturbation
from src.core.types import Sample
from src.perturbations.base import Perturbation


@register_perturbation("label_only")
class LabelOnly(Perturbation):
    """Force output format: letter only, no explanation."""

    @property
    def name(self) -> str:
        return "label_only"

    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        s = deepcopy(sample)
        # Audio is kept — label_only is an output-format constraint
        s.metadata["label_only"] = True
        return s
