#!/usr/bin/env python3
"""Debug MOSS-Audio inference error."""
import sys, os, traceback, warnings
warnings.filterwarnings("ignore")
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

sys.path.insert(0, "/home/lt/projects/audioQAagent")
sys.path.insert(0, "/home/lt/projects/audioQAagent/third_party")

import torch
import numpy as np
from moss_audio_vendor.modeling_moss_audio import MossAudioModel
from moss_audio_vendor.processing_moss_audio import MossAudioProcessor
from moss_audio_vendor.audio_io import load_audio
from src.data.audio_utils import audio_to_tempfile

MODEL_PATH = "OpenMOSS-Team/MOSS-Audio-8B-Instruct"

def main():
    print("=== Step 1: Loading processor ===", flush=True)
    proc = MossAudioProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print(f"mel_sr={proc.config.mel_sr}, mel_dim={proc.config.mel_dim}, n_fft={proc.config.mel_n_fft}", flush=True)

    print("=== Step 2: Loading model ===", flush=True)
    model = MossAudioModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print("Model loaded successfully.", flush=True)

    # Test with actual inference flow
    print("=== Step 3: Creating test audio ===", flush=True)
    sr = 16000
    duration = 2.0
    test_audio = torch.randn(int(sr * duration), dtype=torch.float32)

    tmp_path = audio_to_tempfile(test_audio, sr=sr)
    raw_audio = load_audio(tmp_path, sample_rate=proc.config.mel_sr)
    os.unlink(tmp_path)
    print(f"Audio shape: {raw_audio.shape}, dtype: {raw_audio.dtype}", flush=True)

    print("=== Step 4: Processor call ===", flush=True)
    prompt = "What sound is this?\nA. Dog\nB. Cat\nC. Bird\nD. Car\nAnswer:"
    inputs = proc(text=prompt, audios=[raw_audio], return_tensors="pt")
    print(f"input_ids: {inputs['input_ids'].shape}, audio_data: {inputs['audio_data'].shape}", flush=True)

    inputs = inputs.to(model.device)
    if inputs.get("audio_data") is not None:
        inputs["audio_data"] = inputs["audio_data"].to(model.dtype)
    inputs["audio_token_id"] = proc.audio_token_id
    audio_tokens = (inputs["input_ids"] == proc.audio_token_id).sum().item()
    print(f"audio_tokens: {audio_tokens}", flush=True)

    print("=== Step 5: Model forward ===", flush=True)
    try:
        with torch.no_grad():
            out = model(**inputs)
        print(f"Forward OK. logits shape: {out.logits.shape}", flush=True)
    except Exception:
        print("Forward FAILED:", flush=True)
        traceback.print_exc()

    print("=== Step 6: Model generate ===", flush=True)
    try:
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16, do_sample=False, use_cache=True)
        decoded = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Generated: {decoded[0].strip()}", flush=True)
    except Exception:
        print("Generate FAILED:", flush=True)
        traceback.print_exc()

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
