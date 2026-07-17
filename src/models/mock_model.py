"""Mock model for pipeline testing without real model weights.

Returns deterministic answers based on text matching, with no audio processing.
Useful for testing the full evaluation pipeline offline.
"""

import random
from typing import Tuple

from src.core.registry import register_model
from src.models.base import AbstractModel
from src.models.utils import clean_answer, normalize_text


@register_model("mock_model")
class MockModel(AbstractModel):
    """A mock model that returns answers based on simple heuristics.

    No real model weights needed. Used for pipeline testing.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self._loaded = False
        self._seed = config.get("seed", 42)

    @property
    def name(self) -> str:
        return "mock_model"

    @property
    def supports_text_only(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Simple heuristic: pick the first choice
        answer = choices[0] if choices else ""
        if label_only:
            raw = "A"  # label-only: just the letter
        else:
            raw = f"<answer>{answer}</answer>"

        return answer, raw

    def unload(self) -> None:
        self._loaded = False
