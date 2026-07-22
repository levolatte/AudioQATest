"""Qwen2-Audio-7B-Instruct model adapter.

Uses Qwen2AudioForConditionalGeneration. Handles audio path and tensor input.
"""

import os
from typing import Tuple

import librosa
import numpy as np
import torch

from src.core.registry import register_model
from src.models.base import AbstractModel, ModelLoadError, ModelInferenceError
from src.models.utils import build_question_prompt, clean_answer


@register_model("qwen2_audio")
class Qwen2AudioModel(AbstractModel):
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

    @property
    def name(self) -> str:
        return "qwen2_audio"

    @property
    def supports_text_only(self) -> bool:
        return False

    def load(self) -> None:
        try:
            from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._model_path,
                trust_remote_code=self._processor_kwargs.get("trust_remote_code", True),
            )

            load_kwargs = {
                "trust_remote_code": self._processor_kwargs.get("trust_remote_code", True),
                "device_map": self._device_map,
            }

            if self._max_memory:
                load_kwargs["max_memory"] = self._max_memory

            if self._load_in_4bit:
                load_kwargs["load_in_4bit"] = True
            elif self._load_in_8bit:
                load_kwargs["load_in_8bit"] = True
            else:
                if self._dtype == "bfloat16":
                    load_kwargs["torch_dtype"] = torch.bfloat16
                elif self._dtype == "float16":
                    load_kwargs["torch_dtype"] = torch.float16

            self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
                self._model_path,
                **load_kwargs,
            )
            self._model.eval()

        except Exception as e:
            raise ModelLoadError(f"Failed to load Qwen2-Audio: {e}") from e

    def _decode_audio(self, audio) -> np.ndarray:
        """Decode audio input to a 1-D float32 numpy array at the
        processor's expected sampling rate.  Accepts file paths and
        torch tensors; rejects everything else.
        """
        if isinstance(audio, torch.Tensor):
            arr = audio.cpu().numpy()
        elif isinstance(audio, str):
            if not os.path.exists(audio):
                raise FileNotFoundError(f"Audio file not found: {audio}")
            sr = self._processor.feature_extractor.sampling_rate
            arr, _ = librosa.load(audio, sr=sr, mono=True)
        else:
            raise TypeError(
                f"Unsupported audio type: {type(audio).__name__}. "
                f"Expected str (path) or torch.Tensor (waveform)."
            )

        # --- Normalise to 1-D float32 -------------------------------------
        if arr.ndim != 1:
            arr = arr.squeeze()
        if arr.ndim != 1:
            raise ValueError(
                f"Audio waveform must be 1-D after squeeze, got shape {arr.shape}"
            )
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        if len(arr) == 0:
            raise ValueError("Decoded audio waveform is empty")
        if not np.isfinite(arr).all():
            raise ValueError("Audio waveform contains NaN or Inf values")

        return arr

    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        try:
            # --- Validate & decode audio ------------------------------------
            if audio is None:
                raise ValueError(
                    "Qwen2-Audio requires audio input (supports_text_only=False), got None"
                )

            waveform = self._decode_audio(audio)

            # --- Build messages and tokenize --------------------------------
            prompt = build_question_prompt(question, choices, label_only=label_only)

            audio_url = audio if isinstance(audio, str) else "audio.wav"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_url},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            # Official Qwen2-Audio processor signature (transformers 5.x):
            #   __call__(text=..., audio=np.ndarray|list[np.ndarray]=None)
            # audio= receives flat list of 1-D float32 numpy arrays
            # decoded at the feature extractor's sampling rate.
            inputs = self._processor(
                text=text,
                audio=[waveform],
                sampling_rate=self._processor.feature_extractor.sampling_rate,
                return_tensors="pt",
                padding=True,
            )

            # --- Post-processor assertions ----------------------------------
            required = {"input_ids", "attention_mask", "input_features"}
            missing = required - set(inputs.keys())
            if missing:
                raise RuntimeError(
                    f"Missing model inputs after processor: {sorted(missing)}"
                )

            features = inputs["input_features"]
            if features.numel() == 0:
                raise RuntimeError("input_features is empty after processor")
            if not torch.isfinite(features).all():
                raise RuntimeError("input_features contains NaN or Inf values")

            inputs = inputs.to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs)

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]

            pred_raw = self._processor.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            pred = clean_answer(pred_raw, choices)

            return pred, pred_raw

        except Exception as e:
            raise ModelInferenceError(f"Qwen2-Audio inference failed: {e}") from e

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """Batch inference for Qwen2-Audio.

        Official API: every audio is decoded to a 1-D float32 numpy
        array, then passed as a *flat* list to ``audios=``.
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        all_texts: list[str] = []
        all_waveforms: list[np.ndarray] = []
        all_choices: list[list[str]] = []

        for audio, question, choices, label_only in batch:
            if audio is None:
                raise ValueError(
                    "Qwen2-Audio requires audio input (supports_text_only=False), got None"
                )

            prompt = build_question_prompt(question, choices, label_only=label_only)
            waveform = self._decode_audio(audio)

            audio_url = audio if isinstance(audio, str) else "audio.wav"
            messages = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_url},
                    {"type": "text", "text": prompt},
                ]},
            ]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            all_texts.append(text)
            all_waveforms.append(waveform)
            all_choices.append(choices)

        # --- Safety assertion ------------------------------------------------
        if len(all_texts) != len(all_waveforms):
            raise RuntimeError(
                f"Text/waveform count mismatch: "
                f"{len(all_texts)} vs {len(all_waveforms)}"
            )

        try:
            # Official Qwen2-Audio processor: flat list of 1-D float32 numpy arrays
            inputs = self._processor(
                text=all_texts,
                audio=all_waveforms,
                sampling_rate=self._processor.feature_extractor.sampling_rate,
                padding=True,
                return_tensors="pt",
            )

            # --- Post-processor assertions ----------------------------------
            required = {"input_ids", "attention_mask", "input_features"}
            missing = required - set(inputs.keys())
            if missing:
                raise RuntimeError(
                    f"Missing model inputs after processor: {sorted(missing)}"
                )

            features = inputs["input_features"]
            if features.numel() == 0:
                raise RuntimeError("input_features is empty after processor")
            if not torch.isfinite(features).all():
                raise RuntimeError("input_features contains NaN or Inf values")

            inputs = inputs.to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs)

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]

            decoded = self._processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )

            results: list[tuple[str, str]] = []
            for j, raw_text in enumerate(decoded):
                raw_text = raw_text.strip()
                chosen = clean_answer(raw_text, all_choices[j])
                results.append((chosen, raw_text))
            return results

        except Exception as e:
            raise ModelInferenceError(f"Qwen2-Audio batch inference failed: {e}") from e

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
