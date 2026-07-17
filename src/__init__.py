"""Audio QA Agent - Evaluation framework for audio understanding models."""

import os

# ── HuggingFace mirror ──────────────────────────────────────────────
# Redirect all HF downloads (models, processors, datasets) through a mirror.
# Set the HF_ENDPOINT env var before running to override:
#   HF_ENDPOINT=https://hf-mirror.com python scripts/run_experiment.py ...
# Or leave unset to default to the mirror configured below.
_HF_MIRROR = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["HF_ENDPOINT"] = _HF_MIRROR

# ── HuggingFace token ───────────────────────────────────────────────
# Used for gated models (e.g. Qwen2.5-Omni-7B) and higher rate limits.
# Set via HF_TOKEN env var or configure hf_token in configs/base.yaml.
_HF_TOKEN = os.environ.get("HF_TOKEN", "")
if _HF_TOKEN:
    os.environ["HF_TOKEN"] = _HF_TOKEN
