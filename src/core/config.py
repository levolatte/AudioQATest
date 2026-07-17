"""Configuration loading, merging, and validation via Pydantic.

Configuration precedence (highest last):
    base.yaml  ->  model.yaml  ->  benchmark.yaml  ->  experiment.yaml
"""

import os
import copy
from pathlib import Path
from typing import Optional

import yaml

from src.core.types import (
    ModelConfig,
    BenchmarkConfig,
    PerturbationConfig,
    RuntimeConfig,
    ExperimentConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override dict into base dict."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _load_yaml(path: Path) -> dict:
    """Load a YAML file, return empty dict if not found."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _apply_hf_settings(config: dict) -> None:
    """Set HF_ENDPOINT / HF_TOKEN env vars.

    Precedence (user env var > config YAML):
      1. If ``HF_TOKEN`` is already set in ``os.environ``, keep it.
      2. Otherwise fall back to ``hf_token`` from config YAML.
    """
    hf_endpoint = config.get("hf_endpoint", "")
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint

    # Only read from YAML if the env var hasn't been set externally
    if "HF_TOKEN" not in os.environ:
        hf_token = config.get("hf_token", "")
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token


def load_config(experiment_name: str) -> ExperimentConfig:
    """Load a full experiment configuration by name.

    Looks for configs/experiments/{experiment_name}.yaml and merges with
    base, model, and benchmark configs referenced within.
    """
    # 1. Load base config
    base_raw = _load_yaml(CONFIG_DIR / "base.yaml")

    # 2. Apply HF settings early (before any model/dataset downloads)
    _apply_hf_settings(base_raw)

    # 3. Load experiment config
    exp_path = CONFIG_DIR / "experiments" / f"{experiment_name}.yaml"
    exp_raw = _load_yaml(exp_path)
    if not exp_raw:
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")

    # 4. Merge base -> experiment
    merged = _deep_merge(base_raw, exp_raw)
    _apply_hf_settings(merged)  # experiment-level override

    experiment = ExperimentConfig(
        name=merged.get("experiment", {}).get("name", experiment_name),
        description=merged.get("experiment", {}).get("description", ""),
        seed=merged.get("seed", 42),
        output_dir=merged.get("output_dir", merged.get("output_base_dir", "outputs")),
        models=merged.get("models", []),
        benchmarks=merged.get("benchmarks", []),
        perturbations=merged.get("perturbations", []),
        runtime=RuntimeConfig(**merged.get("runtime", {})),
    )

    return experiment


def load_model_config(model_name: str) -> ModelConfig:
    """Load a single model configuration by name."""
    path = CONFIG_DIR / "models" / f"{model_name}.yaml"
    data = _load_yaml(path)
    if not data:
        raise FileNotFoundError(f"Model config not found: {path}")
    return ModelConfig(**data)


def load_benchmark_config(benchmark_name: str) -> BenchmarkConfig:
    """Load a single benchmark configuration by name."""
    path = CONFIG_DIR / "benchmarks" / f"{benchmark_name}.yaml"
    data = _load_yaml(path)
    if not data:
        raise FileNotFoundError(f"Benchmark config not found: {path}")
    return BenchmarkConfig(**data)


def resolve_perturbations(perturbation_names: list[str]) -> list[PerturbationConfig]:
    """Resolve perturbation name strings to PerturbationConfig objects.

    Names are passed through as-is (no special aliasing).
    """
    result = []
    for name in perturbation_names:
        result.append(PerturbationConfig(name=name))
    return result
