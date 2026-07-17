"""Perturbation conditions for robustness evaluation.

Each perturbation is auto-registered via the @register_perturbation decorator.
Import modules to trigger registration.
"""

from src.perturbations.base import Perturbation

# Import to trigger registration
from src.perturbations import baseline          # noqa: F401
from src.perturbations import silent_audio      # noqa: F401
from src.perturbations import noise_audio       # noqa: F401
from src.perturbations import shuffled_choices  # noqa: F401
from src.perturbations import label_only        # noqa: F401
