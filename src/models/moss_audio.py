"""MOSS-Audio model adapter.

MOSS-Audio by Fudan University's OpenMOSS team.
Available size: 8B (MOSS-Audio-8B-Instruct).

Architecture: MOSS-Audio-Encoder + Qwen3-LLM backbone.
Uses vendored copy of MOSS-Audio source under third_party/moss_audio_vendor/
to avoid 'src' namespace conflict with the project itself.
"""

import os
import sys
from pathlib import Path
from typing import Tuple

import torch

from src.core.registry import register_model
from src.models.base import AbstractModel, ModelLoadError, ModelInferenceError
from src.models.utils import build_question_prompt, clean_answer

# Path to third_party directory (parent of moss_audio_vendor)
_THIRD_PARTY = str(
    Path(__file__).resolve().parent.parent.parent / "third_party"
)


@register_model("moss_audio")
class MOSSAudioModel(AbstractModel):
    def __init__(self, config: dict = None):
        config = config or {}
        self._model_path = config.get("path", config.get("local_path", config.get("hf_model_id", "")))
        self._dtype = config.get("dtype", "bfloat16")
        self._device_map = config.get("device_map", "auto")
        self._load_in_4bit = config.get("load_in_4bit", False)
        self._load_in_8bit = config.get("load_in_8bit", False)
        self._processor_kwargs = config.get("processor_kwargs", {})
        self._generate_kwargs = config.get("generate_kwargs", {})
        self._max_memory = config.get("max_memory", None)
        self._model = None
        self._processor = None
        self._model_cls = None
        self._processor_cls = None
        self._load_audio_fn = None

    @property
    def name(self) -> str:
        return "moss_audio"

    @property
    def supports_text_only(self) -> bool:
        return False

    def _ensure_imports(self):
        if self._model_cls is not None:
            return
        # Add third_party to path so moss_audio_vendor is importable
        if _THIRD_PARTY not in sys.path:
            sys.path.insert(0, _THIRD_PARTY)

        from moss_audio_vendor.modeling_moss_audio import MossAudioModel as _ModelCls
        from moss_audio_vendor.processing_moss_audio import MossAudioProcessor as _ProcCls
        from moss_audio_vendor.audio_io import load_audio as _load_audio

        self._model_cls = _ModelCls
        self._processor_cls = _ProcCls
        self._load_audio_fn = _load_audio

    def load(self) -> None:
        try:
            self._ensure_imports()

            trust_remote = self._processor_kwargs.get("trust_remote_code", True)

            self._processor = self._processor_cls.from_pretrained(
                self._model_path,
                trust_remote_code=trust_remote,
            )

            load_kwargs = {
                "trust_remote_code": trust_remote,
            }

            if self._load_in_4bit:
                load_kwargs["load_in_4bit"] = True
            elif self._load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            else:
                dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16}
                if self._dtype in dtype_map:
                    load_kwargs["torch_dtype"] = dtype_map[self._dtype]
                elif self._dtype == "auto":
                    load_kwargs["torch_dtype"] = "auto"

            if self._device_map:
                load_kwargs["device_map"] = self._device_map
            if self._max_memory:
                load_kwargs["max_memory"] = self._max_memory

            self._model = self._model_cls.from_pretrained(
                self._model_path,
                **load_kwargs,
            )
            self._model.eval()

        except Exception as e:
            raise ModelLoadError(f"Failed to load MOSS-Audio: {e}") from e

    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _cleanup_temp = False
        audio_path = None

        try:
            prompt = build_question_prompt(question, choices, label_only=label_only)

            if audio is not None:
                if isinstance(audio, torch.Tensor):
                    audio_path = self._tensor_to_tempfile(audio)
                    _cleanup_temp = True
                else:
                    audio_path = audio

                raw_audio = self._load_audio_fn(
                    audio_path,
                    sample_rate=self._processor.config.mel_sr,
                )
                audio_list = [raw_audio]
            else:
                audio_list = []

            inputs = self._processor(
                text=prompt,
                audios=audio_list if audio_list else None,
                return_tensors="pt",
            )
            inputs = inputs.to(self._model.device)

            if inputs.get("audio_data") is not None:
                inputs["audio_data"] = inputs["audio_data"].to(self._model.dtype)

            inputs["audio_token_id"] = self._processor.audio_token_id

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    **self._generate_kwargs,
                    use_cache=True,
                )

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]

            decoded = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            pred_raw = decoded[0].strip() if decoded else ""

            pred = clean_answer(pred_raw, choices)

            return pred, pred_raw

        except Exception as e:
            raise ModelInferenceError(f"MOSS-Audio inference failed: {e}") from e

        finally:
            # Cleanup temp file — always runs, even on exception
            if _cleanup_temp and audio_path is not None:
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """Sequential inference — MOSS-Audio only supports batch_size=1.

        The WhisperFeatureExtractor uses CPU mel computation and the
        WhisperEncoder self-attention is O(n²), so batching degrades throughput.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results: list[tuple[str, str]] = []
        for audio, question, choices, label_only in batch:
            try:
                chosen, raw = self.infer(audio, question, choices, label_only=label_only)
            except Exception:
                chosen = choices[0] if choices else ""
                raw = "[ERROR]"
            results.append((chosen, raw))
        return results

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()


# Register size-specific aliases so config names match
register_model("moss_audio_8b")(MOSSAudioModel)

