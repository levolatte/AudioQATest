"""Shuffled choices perturbation: randomly permute multiple-choice option order.

Key design: the ground truth is REMAPPED to its new position after shuffling,
so the downstream evaluation logic doesn't need to know about the shuffle.
The perturbation records the original order in metadata["original_choices"]
for traceability.
"""

import random
from copy import deepcopy

from src.core.registry import register_perturbation
from src.core.types import Sample
from src.perturbations.base import Perturbation


@register_perturbation("shuffled_choices")
class ShuffledChoices(Perturbation):
    """Randomly permute the order of multiple-choice options.

    The ground truth answer is remapped to its new position in the shuffled
    list so that correctness evaluation works normally.
    """

    @property
    def name(self) -> str:
        return "shuffled_choices"

    def apply(self, sample: Sample, rng: random.Random, **kwargs) -> Sample:
        s = deepcopy(sample)

        if len(s.choices) <= 1:
            return s

        original_choices = list(s.choices)
        original_answer = s.ground_truth

        # Build permutation
        indices = list(range(len(s.choices)))
        rng.shuffle(indices)

        s.choices = [original_choices[i] for i in indices]
        s.metadata["original_choices"] = original_choices
        s.metadata["shuffle_indices"] = indices

        # Remap ground truth: find its new position
        # Match original answer text to original choices
        from src.models.utils import normalize_text
        answer_norm = normalize_text(original_answer)

        for i, choice in enumerate(original_choices):
            if normalize_text(choice) == answer_norm:
                # Found it in original list. Its new position is where i maps in indices
                new_index = indices.index(i)
                s.ground_truth = s.choices[new_index]
                break

        return s
