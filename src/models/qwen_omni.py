"""Qwen2.5-Omni-7B model adapter.

Refactored from the original infer.py. Handles:
- Audio as file path or in-memory tensor (via temp file)
- Multi-modal messages format (OpenAI-style)
- qwen_omni_utils process_mm_info for audio extraction
"""

import os
import tempfile
from typing import Tuple, Union, Optional

import torch

from src.core.registry import register_model
from src.models.base import AbstractModel, ModelLoadError, ModelInferenceError
from src.models.utils import build_question_prompt, clean_answer

@register_model("qwen_omni")
class QwenOmniModel(AbstractModel):
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
        return "qwen_omni"

    @property
    def supports_text_only(self) -> bool:
        return True

    def load(self) -> None:
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor
            from qwen_omni_utils import process_mm_info  # noqa: F401 — used in infer()

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
                dtype_str = self._dtype
                if dtype_str == "bfloat16":
                    load_kwargs["torch_dtype"] = torch.bfloat16
                elif dtype_str == "float16":
                    load_kwargs["torch_dtype"] = torch.float16

            self._model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
                self._model_path,
                **load_kwargs,
            )
            self._model.eval()

        except Exception as e:
            raise ModelLoadError(f"Failed to load Qwen2.5-Omni: {e}") from e

    def _build_single_messages(self, audio, prompt: str) -> tuple[list[dict], bool, str]:
        """Build messages list for one sample and handle tensor→tempfile.

        Returns (messages, cleanup_temp, audio_path_or_data).
        """
        from qwen_omni_utils import process_mm_info  # noqa: F401 — used below

        _cleanup_temp = False
        audio_path_or_data = None

        user_content: list[dict] = []

        if audio is not None:
            if isinstance(audio, torch.Tensor):
                audio_path_or_data = self._tensor_to_tempfile(audio)
                _cleanup_temp = True
            else:
                audio_path_or_data = audio
            user_content.append({"type": "audio", "audio": audio_path_or_data})

        user_content.append({"type": "text", "text": prompt})

        messages = [
            {"role": "user", "content": user_content},
        ]
        return messages, _cleanup_temp, audio_path_or_data

    def infer(self, audio, question: str,
              choices: list[str], label_only: bool = False) -> Tuple[str, str]:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        prompt = build_question_prompt(question, choices, label_only=label_only)
        messages, _cleanup_temp, audio_path_or_data = self._build_single_messages(audio, prompt)

        try:
            from qwen_omni_utils import process_mm_info

            text = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )

            audios, images, videos = process_mm_info(
                messages,
                use_audio_in_video=self._processor_kwargs.get("use_audio_in_video", False),
            )

            inputs = self._processor(
                text=text, audio=audios, images=images, videos=videos,
                return_tensors="pt", padding=True,
                use_audio_in_video=self._processor_kwargs.get("use_audio_in_video", False),
            )
            inputs = inputs.to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, **self._generate_kwargs)

            input_len = inputs["input_ids"].shape[1]
            output_ids = generated_ids[:, input_len:]

            pred_raw = self._processor.batch_decode(
                output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
            )[0].strip()

            pred = clean_answer(pred_raw, choices)
            return pred, pred_raw

        except Exception as e:
            raise ModelInferenceError(f"Inference failed: {e}") from e
        finally:
            if _cleanup_temp and audio_path_or_data is not None:
                try:
                    os.unlink(audio_path_or_data)
                except OSError:
                    pass

    def infer_batch(self, batch: list[tuple]) -> list[tuple[str, str]]:
        """Batch inference for Qwen Omni — multiple samples in one generate()."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        from qwen_omni_utils import process_mm_info

        # Build all messages / prompts / temp-file tracking
        all_messages: list[list[dict]] = []
        all_choices: list[list[str]] = []
        cleanup_items: list[tuple[bool, str]] = []  # (_cleanup, path_or_none)

        for audio, question, choices, label_only in batch:
            prompt = build_question_prompt(question, choices, label_only=label_only)
            messages, _cleanup_temp, audio_path_or_data = self._build_single_messages(audio, prompt)
            all_messages.append(messages)
            all_choices.append(choices)
            cleanup_items.append((_cleanup_temp, audio_path_or_data))

        try:
            # Apply chat template per sample (each has a different prompt)
            texts = []
            for messages in all_messages:
                text = self._processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                texts.append(text)

            # Extract ALL media at once — process_mm_info handles a list of
            # conversations natively and returns flat lists (one entry per
            # sample).  Calling it per-sample and nesting is what caused the
            # broadcast error.
            use_audio_in_video = self._processor_kwargs.get("use_audio_in_video", False)
            audios, images, videos = process_mm_info(all_messages, use_audio_in_video=use_audio_in_video)

            # Processor handles padding across the batch automatically
            inputs = self._processor(
                text=texts,
                audio=audios,
                images=images if images else None,
                videos=videos if videos else None,
                return_tensors="pt",
                padding=True,
                use_audio_in_video=use_audio_in_video,
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
            raise ModelInferenceError(f"Batch inference failed: {e}") from e
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
