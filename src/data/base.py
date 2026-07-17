"""Abstract benchmark interface and Sample dataclass."""

from abc import ABC, abstractmethod

from src.core.types import Sample


class AbstractBenchmark(ABC):
    """Interface for benchmark dataset loaders.

    Each benchmark (MMAU, MMAR, etc.) implements this interface to provide
    a uniform iterator of Sample objects.
    """

    @abstractmethod
    def load(self, split: str) -> None:
        """Load the specified split into memory."""
        ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, idx: int) -> Sample: ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark identifier (e.g. 'mmau', 'mmar')."""
        ...

    @property
    @abstractmethod
    def category_fields(self) -> list[str]:
        """Metadata keys available for per-category accuracy breakdown."""
        ...

    @property
    def audio_samples(self) -> list:
        """Return list of audio references for use by unrelated-audio perturbation.

        Subclasses should override to return lightweight references
        (paths, tensors, or indices) that can be used to retrieve audio data.
        """
        return []
