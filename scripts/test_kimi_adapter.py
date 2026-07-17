#!/usr/bin/env python3
"""Quick test of the KimiAudioModel adapter (via official KimiAudio API)."""
import os, sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.kimi_audio import KimiAudioModel

config = {
    "hf_model_id": "moonshotai/Kimi-Audio-7B-Instruct",
    "local_path": "/home/lt/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b",
    "generate_kwargs": {
        "max_new_tokens": 10,
    },
}

print("Loading model via adapter...", flush=True)
model = KimiAudioModel(config=config)
model.load()
print("Model loaded!", flush=True)

# Test 1: Text-only inference (label_only mode, no audio)
print("\nTest 1: Text-only (correct answer = C)...", flush=True)
pred, raw = model.infer(
    audio=None,
    question="What is the largest planet in our solar system?",
    choices=["Earth", "Mars", "Jupiter", "Venus"],
    label_only=True,
)
print(f"  Raw output: {repr(raw)}", flush=True)
print(f"  Predicted:  {pred}", flush=True)

# Test 2: Try with different label order
print("\nTest 2: Text-only (correct answer = B)...", flush=True)
pred2, raw2 = model.infer(
    audio=None,
    question="What is 2+2?",
    choices=["5", "4", "3", "6"],
    label_only=True,
)
print(f"  Raw output: {repr(raw2)}", flush=True)
print(f"  Predicted:  {pred2}", flush=True)

model.unload()
print("\nDone!", flush=True)
