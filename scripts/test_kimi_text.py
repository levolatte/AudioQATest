#!/usr/bin/env python3
"""Quick test of KimiAudio text-only inference via the official API.

This script tests the low-level KimiAudio API directly (not the adapter).
For adapter testing, use test_kimi_adapter.py.
"""
import sys, types, importlib, os
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "third_party" / "Kimi-Audio"))
from kimia_infer.flash_attn_mock import *

def make_mock(name):
    spec = importlib.machinery.ModuleSpec(name, None)
    mod = types.ModuleType(name)
    mod.__spec__ = spec; mod.__file__ = '(mock)'; return mod

fa = make_mock('flash_attn')
fa.flash_attn_func = flash_attn_func
fa.flash_attn_varlen_func = flash_attn_varlen_func
fa.flash_attn_qkvpacked_func = flash_attn_qkvpacked_func
fa.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
sys.modules['flash_attn'] = fa

fa_bp = make_mock('flash_attn.bert_padding')
fa_bp.index_first_axis = index_first_axis; fa_bp.unpad_input = unpad_input; fa_bp.pad_input = pad_input
sys.modules['flash_attn.bert_padding'] = fa_bp

fa3 = make_mock('flash_attn_interface')
fa3.flash_attn_func = flash_attn_func; fa3.flash_attn_varlen_func = flash_attn_varlen_func
sys.modules['flash_attn_interface'] = fa3

from kimia_infer.api.kimia import KimiAudio

MODEL_PATH = '/home/lt/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/snapshots/9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b'
print("Loading model...", flush=True)
model = KimiAudio(model_path=MODEL_PATH, load_detokenizer=False)
print("Model loaded!", flush=True)

print("\nTesting text-only inference...", flush=True)
messages = [{'role': 'user', 'message_type': 'text', 'content': 'What is 2+2? Answer in one word.'}]
wav, text = model.generate(messages, output_type='text', text_temperature=0.0, max_new_tokens=10)
print(f"Response: {repr(text)}", flush=True)
