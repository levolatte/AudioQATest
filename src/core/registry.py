"""Decorator-based registration system for models, benchmarks, and perturbations.

Usage:
    @register_model("qwen_omni")
    class QwenOmniModel(AbstractModel): ...

    @register_benchmark("mmau")
    class MMAUBenchmark(AbstractBenchmark): ...

    @register_perturbation("silent_audio")
    class SilentAudio(Perturbation): ...
"""

from typing import Type, Any

_MODEL_REGISTRY: dict[str, Type[Any]] = {}
_BENCHMARK_REGISTRY: dict[str, Type[Any]] = {}
_PERTURBATION_REGISTRY: dict[str, Type[Any]] = {}


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def register_model(name: str):
    """Decorator to register a model adapter class under a name."""
    def decorator(cls):
        _MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_model(name: str) -> Type[Any]:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model: '{name}'. Registered: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

def register_benchmark(name: str):
    """Decorator to register a benchmark loader class under a name."""
    def decorator(cls):
        _BENCHMARK_REGISTRY[name] = cls
        return cls
    return decorator


def get_benchmark(name: str) -> Type[Any]:
    if name not in _BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark: '{name}'. Registered: {list(_BENCHMARK_REGISTRY)}")
    return _BENCHMARK_REGISTRY[name]


def list_benchmarks() -> list[str]:
    return sorted(_BENCHMARK_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Perturbation registry
# ---------------------------------------------------------------------------

def register_perturbation(name: str):
    """Decorator to register a perturbation class under a name."""
    def decorator(cls):
        _PERTURBATION_REGISTRY[name] = cls
        return cls
    return decorator


def get_perturbation(name: str) -> Type[Any]:
    if name not in _PERTURBATION_REGISTRY:
        raise KeyError(f"Unknown perturbation: '{name}'. Registered: {list(_PERTURBATION_REGISTRY)}")
    return _PERTURBATION_REGISTRY[name]


def list_perturbations() -> list[str]:
    return sorted(_PERTURBATION_REGISTRY.keys())
