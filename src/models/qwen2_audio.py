"""Qwen2-Audio-7B-Instruct model adapter.

Uses Qwen2AudioForConditionalGeneration. Handles audio path and tensor input.
"""

import os
from typing import Tuple

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

    def infer(self, audio, question: str, choices: list[str],
              label_only: bool = False) -> Tuple[str, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        _cleanup_temp = False
        audio_path = audio

        try:
            if isinstance(audio, torch.Tensor):
                audio_path = self._tensor_to_tempfile(audio)
                _cleanup_temp = True

            prompt = build_question_prompt(question, choices, label_only=label_only)

            # Qwen2-Audio uses a conversations format
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio_url": audio_path} if audio_path else {},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Filter empty content blocks
            messages[0]["content"] = [c for c in messages[0]["content"] if c]

            text = self._processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self._processor(
                text=text,
                audios=[audio_path] if audio_path and isinstance(audio_path, str) else None,
                return_tensors="pt",
            )
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

        finally:
            # Cleanup temp file — always runs, even on exception
            if _cleanup_temp and isinstance(audio_path, str):
                try:
                    os.unlink(audio_path)
                except OSError:
                    pass

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """Batch inference for Qwen2-Audio."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        all_texts: list[str] = []
        all_audios: list[list[str]] = []
        all_choices: list[list[str]] = []
        cleanup_items: list[tuple[bool, str]] = []

        for audio, question, choices, label_only in batch:
            prompt = build_question_prompt(question, choices, label_only=label_only)

            _cleanup = False
            audio_path = audio
            if isinstance(audio, torch.Tensor):
                audio_path = self._tensor_to_tempfile(audio)
                _cleanup = True

            messages = [
                {"role": "user", "content": [
                    {"type": "audio", "audio_url": audio_path} if audio_path else {},
                    {"type": "text", "text": prompt},
                ]},
            ]
            messages[0]["content"] = [c for c in messages[0]["content"] if c]

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            all_texts.append(text)
            all_audios.append([audio_path] if audio_path and isinstance(audio_path, str) else [])
            all_choices.append(choices)
            cleanup_items.append((_cleanup, audio_path if _cleanup else None))

        try:
            has_audio = any(len(a) > 0 for a in all_audios)
            inputs = self._processor(
                text=all_texts,
                audios=all_audios if has_audio else None,
                padding=True,
                return_tensors="pt",
            )
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
        finally:
            for _cleanup, path in cleanup_items:
                if _cleanup and path is not None:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        torch.cuda.empty_cache()
