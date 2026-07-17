"""Kimi-Audio-7B-Instruct model adapter.

Uses the official kimi-audio inference API from third_party/Kimi-Audio/
instead of standard transformers AutoModel + AutoProcessor (unsupported).

Kimi-Audio has a dual-stream architecture:
- Audio stream: discrete speech tokens + continuous whisper features
- Text stream: standard text tokens
- 28 main decoder layers + 6 MIMO (audio) layers
- Two output heads: lm_head (text) and mimo_output (audio)

Flash-attention is replaced with SDPA-based mocks (no nvcc available).
"""

import os
import sys
import types
import importlib
from typing import Tuple

import torch

from src.core.registry import register_model
from src.models.base import AbstractModel, ModelLoadError, ModelInferenceError
from src.models.utils import build_question_prompt, clean_answer

# ── flash_attn mock injection ────────────────────────────────────────────────
# Must happen BEFORE any import of KimiAudio (which transitively imports
# flash_attn via modeling_moonshot_kimia.py and modeling_whisper.py).

_flash_attn_injected = False


def _inject_flash_attn_mock():
    """Inject SDPA-based flash_attn replacement into sys.modules."""
    global _flash_attn_injected
    if _flash_attn_injected:
        return
    _flash_attn_injected = True

    _kimi_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "third_party", "Kimi-Audio"
    ))
    if _kimi_path not in sys.path:
        sys.path.insert(0, _kimi_path)
    from kimia_infer.flash_attn_mock import (
        flash_attn_func, flash_attn_varlen_func,
        flash_attn_qkvpacked_func, flash_attn_varlen_qkvpacked_func,
        index_first_axis, unpad_input, pad_input,
        _fa3_flash_attn_func, _fa3_flash_attn_varlen_func,
    )

    def _make_mod(name):
        spec = importlib.machinery.ModuleSpec(name, None)
        mod = types.ModuleType(name)
        mod.__spec__ = spec
        mod.__file__ = "(mock)"
        return mod

    fa = _make_mod("flash_attn")
    fa.flash_attn_func = flash_attn_func
    fa.flash_attn_varlen_func = flash_attn_varlen_func
    fa.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
    fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    sys.modules["flash_attn"] = fa

    fa_bp = _make_mod("flash_attn.bert_padding")
    fa_bp.index_first_axis = index_first_axis
    fa_bp.unpad_input = unpad_input
    fa_bp.pad_input = pad_input
    sys.modules["flash_attn.bert_padding"] = fa_bp

    fa3 = _make_mod("flash_attn_interface")
    fa3.flash_attn_func = _fa3_flash_attn_func
    fa3.flash_attn_varlen_func = _fa3_flash_attn_varlen_func
    sys.modules["flash_attn_interface"] = fa3


@register_model("kimi_audio")
class KimiAudioModel(AbstractModel):
    def __init__(self, config: dict = None):
        config = config or {}
        # Prefer explicit path, then non-empty local_path, then hf_model_id
        self._model_path = (
            config.get("path") or config.get("local_path") or config.get("hf_model_id", "")
        )
        self._generate_kwargs = config.get("generate_kwargs", {})
        if "max_new_tokens" not in self._generate_kwargs:
            self._generate_kwargs["max_new_tokens"] = 64
        self._model = None

    @property
    def name(self) -> str:
        return "kimi_audio"

    @property
    def supports_text_only(self) -> bool:
        return False  # audio stream always needed for dual-stream architecture

    def load(self) -> None:
        try:
            _inject_flash_attn_mock()
            from kimia_infer.api.kimia import KimiAudio  # noqa: E402

            # Resolve local cache path (offline environment)
            model_path = self._model_path
            if not os.path.exists(model_path):
                model_path = self._resolve_local_path(model_path)

            self._model = KimiAudio(
                model_path=model_path,
                load_detokenizer=False,  # text-only QA, no audio generation
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load Kimi-Audio: {e}") from e

    @staticmethod
    def _resolve_local_path(model_id: str) -> str:
        """Resolve a HF model ID to its local cache snapshot (offline)."""
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        repo_dir = model_id.replace("/", "--") if "/" in model_id else model_id
        # First try: symlink in snapshots/
        refs_dir = os.path.join(cache_dir, f"models--{repo_dir}", "refs")
        if os.path.isdir(refs_dir):
            refs = os.listdir(refs_dir)
            if refs:
                # Read the symlink target
                ref_path = os.path.join(refs_dir, refs[0])
                if os.path.islink(ref_path):
                    resolved = os.readlink(ref_path)
                    return os.path.join(os.path.dirname(refs_dir), "snapshots", resolved)
        # Second try: list snapshots directory
        snaps_dir = os.path.join(cache_dir, f"models--{repo_dir}", "snapshots")
        if os.path.isdir(snaps_dir):
            snaps = sorted(os.listdir(snaps_dir))
            if snaps:
                return os.path.join(snaps_dir, snaps[-1])
        raise FileNotFoundError(
            f"Local cache not found for {model_id}. "
            f"Set local_path in config to the snapshot directory."
        )

    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _cleanup_temp = False
        audio_signal = audio

        try:
            if isinstance(audio, torch.Tensor):
                audio_signal = self._tensor_to_tempfile(audio)
                _cleanup_temp = True

            prompt = build_question_prompt(question, choices, label_only=label_only)

            # Build KimiAudio message format.
            # CRITICAL: text MUST come before audio so that the audio message
            # is the last user message. This ensures has_ct_token=True in
            # get_prompt(), which appends kimia_speech_ct_id to the audio
            # stream. Without this token, the MIMO cross-attention layers
            # never attend to the whisper features, and the model cannot
            # understand the audio.
            # Ref: third_party/Kimi-Audio/infer.py (text-first, audio-second).
            messages = [{
                "role": "user",
                "message_type": "text",
                "content": prompt,
            }]
            if audio_signal is not None and isinstance(audio_signal, str) and os.path.exists(audio_signal):
                messages.append({
                    "role": "user",
                    "message_type": "audio",
                    "content": audio_signal,
                })

            max_new_tokens = self._generate_kwargs.get("max_new_tokens", 64)

            _, raw_output = self._model.generate(
                messages,
                output_type="text",
                text_temperature=0.0,
                text_top_k=1,
                max_new_tokens=max_new_tokens,
            )

            pred = clean_answer(raw_output, choices)
            return pred, (raw_output or "")

        except Exception as e:
            raise ModelInferenceError(f"Kimi-Audio inference failed: {e}") from e

        finally:
            if _cleanup_temp and isinstance(audio_signal, str):
                try:
                    os.unlink(audio_signal)
                except OSError:
                    pass

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """KimiAudio API does not support batching — sequential fallback."""
        return super().infer_batch(batch)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        torch.cuda.empty_cache()
