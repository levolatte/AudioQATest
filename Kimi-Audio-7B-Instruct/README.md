---
license: mit
language:
- en
- zh
tags:
- audio
- audio-language-model
- speech-recognition
- audio-understanding
- text-to-speech
- audio-generation
- chat
library_name: kimi-audio
---

# Kimi-Audio

<p align="center">
    <img src="https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/assets/kimia_logo.png" width="400"/> <!-- TODO: Replace with actual raw image URL from your repo -->
<p>

<p align="center">
<a href="https://huggingface.co/moonshotai/Kimi-Audio-7B">🤗 Kimi-Audio-7B</a>&nbsp; | <a href="https://huggingface.co/moonshotai/Kimi-Audio-7B-Instruct">🤗 Kimi-Audio-7B-Instruct </a>&nbsp; | <a href="https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/assets/kimia_report.pdf">📑 Paper</a>
</p>

## Introduction

We present Kimi-Audio, an open-source audio foundation model excelling in **audio understanding, generation, and conversation**. This repository hosts the model checkpoints for Kimi-Audio-7B-Instruct.

Kimi-Audio is designed as a universal audio foundation model capable of handling a wide variety of audio processing tasks within a single unified framework. Key features include:

*   **Universal Capabilities:** Handles diverse tasks like speech recognition (ASR), audio question answering (AQA), audio captioning (AAC), speech emotion recognition (SER), sound event/scene classification (SEC/ASC) and end-to-end speech conversation.
*   **State-of-the-Art Performance:** Achieves SOTA results on numerous audio benchmarks (see our [Technical Report](https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/assets/kimia_report.pdf)).
*   **Large-Scale Pre-training:** Pre-trained on over 13 million hours of diverse audio data (speech, music, sounds) and text data.
*   **Novel Architecture:** Employs a hybrid audio input (continuous acoustic + discrete semantic tokens) and an LLM core with parallel heads for text and audio token generation.
*   **Efficient Inference:** Features a chunk-wise streaming detokenizer based on flow matching for low-latency audio generation.

For more details, please refer to our [GitHub Repository](https://github.com/MoonshotAI/Kimi-Audio) and [Technical Report](https://raw.githubusercontent.com/MoonshotAI/Kimi-Audio/master/assets/kimia_report.pdf).

## Requirements

We recommend that you build a Docker image to run the inference. After cloning the inference code, you can construct the image using the `docker build` command.
```bash
git clone https://github.com/MoonshotAI/Kimi-Audio
git submodule update --init
cd Kimi-Audio
docker build -t kimi-audio:v0.1 .
```
Alternatively, You can also use our pre-built image:
```bash
docker pull moonshotai/kimi-audio:v0.1
```

Or, you can install requirments by:
```bash
pip install -r requirements.txt
```

You may refer to the Dockerfile in case of any environment issues.

## Quickstart

This example demonstrates basic usage for generating text from audio (ASR) and generating both text and speech in a conversational turn using the `Kimi-Audio-7B-Instruct` model.

```python
import soundfile as sf
# Assuming the KimiAudio class is available after installation
from kimia_infer.api.kimia import KimiAudio
import torch # Ensure torch is imported if needed for device placement

# --- 1. Load Model ---
# Load the model from Hugging Face Hub
# Make sure you are logged in (`huggingface-cli login`) if the repo is private.
model_id = "moonshotai/Kimi-Audio-7B-Instruct" # Or "Kimi/Kimi-Audio-7B"
device = "cuda" if torch.cuda.is_available() else "cpu" # Example device placement
# Note: The KimiAudio class might handle model loading differently.
# You might need to pass the model_id directly or download checkpoints manually
# and provide the local path as shown in the original readme_kimia.md.
# Please refer to the main Kimi-Audio repository for precise loading instructions.
# Example assuming KimiAudio takes the HF ID or a local path:
try:
    model = KimiAudio(model_path=model_id, load_detokenizer=True) # May need device argument
    model.to(device) # Example device placement
except Exception as e:
    print(f"Automatic loading from HF Hub might require specific setup.")
    print(f"Refer to Kimi-Audio docs. Trying local path example (update path!). Error: {e}")
    # Fallback example:
    # model_path = "/path/to/your/downloaded/kimia-hf-ckpt" # IMPORTANT: Update this path if loading locally
    # model = KimiAudio(model_path=model_path, load_detokenizer=True)
    # model.to(device) # Example device placement

# --- 2. Define Sampling Parameters ---
sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}

# --- 3. Example 1: Audio-to-Text (ASR) ---
# TODO: Provide actual example audio files or URLs accessible to users
# E.g., download sample files first or use URLs
# wget https://path/to/your/asr_example.wav -O asr_example.wav
# wget https://path/to/your/qa_example.wav -O qa_example.wav
asr_audio_path = "asr_example.wav" # IMPORTANT: Make sure this file exists
qa_audio_path = "qa_example.wav" # IMPORTANT: Make sure this file exists

messages_asr = [
    {"role": "user", "message_type": "text", "content": "Please transcribe the following audio:"},
    {"role": "user", "message_type": "audio", "content": asr_audio_path}
]

# Generate only text output
# Note: Ensure the model object and generate method accept device placement if needed
_, text_output = model.generate(messages_asr, **sampling_params, output_type="text")
print(">>> ASR Output Text: ", text_output)
# Expected output: "这并不是告别，这是一个篇章的结束，也是新篇章的开始。" (Example)

# --- 4. Example 2: Audio-to-Audio/Text Conversation ---
messages_conversation = [
    {"role": "user", "message_type": "audio", "content": qa_audio_path}
]

# Generate both audio and text output
wav_output, text_output = model.generate(messages_conversation, **sampling_params, output_type="both")

# Save the generated audio
output_audio_path = "output_audio.wav"
# Ensure wav_output is on CPU and flattened before saving
sf.write(output_audio_path, wav_output.detach().cpu().view(-1).numpy(), 24000) # Assuming 24kHz output
print(f">>> Conversational Output Audio saved to: {output_audio_path}")
print(">>> Conversational Output Text: ", text_output)
# Expected output: "A." (Example)

print("Kimi-Audio inference examples complete.")

```

## Citation

If you find Kimi-Audio useful in your research or applications, please cite our technical report:

```bibtex
@misc{kimi_audio_2024,
      title={Kimi-Audio Technical Report},
      author={Kimi Team},
      year={2024},
      eprint={arXiv:placeholder},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## License

The model is based and modified from [Qwen 2.5-7B](https://github.com/QwenLM/Qwen2.5). Code derived from Qwen2.5-7B is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). Other parts of the code are licensed under the [MIT License](https://opensource.org/licenses/MIT).
