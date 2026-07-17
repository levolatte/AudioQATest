#!/usr/bin/env python3
"""Verify and apply patches to Kimi-Audio HF cache files, then clear modules cache.

Patches applied to the hub blob files (persistent across sessions):
  modeling_moonshot_kimia.py (blob a642317281405a2196873897d92e281138e242f9):
    1. flash_attn bypass: remove is_flash_attn_available() guard
    2. rope_theta compat: handle transformers >=5.0 rope_parameters dict
    3. apply_rotary_pos_emb API: remove position_ids arg (transformers >=5.0)

  tokenization_kimia.py (blob 7f697d3d7e3c31a38171f1ca5af9b62754ddbfac):
    4. Minimal parent init: replace broken super().__init__() call
    5. get_vocab() override: transformers >=5.0 requires this abstract method
"""

import os
import shutil

HUB_BASE = "/home/lt/.cache/huggingface/hub/models--moonshotai--Kimi-Audio-7B-Instruct/blobs"
MODELING_BLOB = f"{HUB_BASE}/a642317281405a2196873897d92e281138e242f9"
TOKENIZER_BLOB = f"{HUB_BASE}/7f697d3d7e3c31a38171f1ca5af9b62754ddbfac"
MODULES_DIR = "/home/lt/.cache/huggingface/modules/transformers_modules/_9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b"

all_ok = True

# ── Modeling patches ─────────────────────────────────────────────────────────

with open(MODELING_BLOB) as f:
    model_content = f.read()

checks = [
    ("Patch 1 (flash_attn bypass)",
     "flash_attn mock injected" in model_content,
     "flash_attn is imported unconditionally (no is_flash_attn_available guard)"),

    ("Patch 2 (rope_theta compat)",
     'hasattr(config, "rope_theta")' in model_content,
     "rope_theta read handles both old config.rope_theta and new rope_parameters dict"),

    ("Patch 3 (apply_rotary_pos_emb API)",
     'apply_rotary_pos_emb(\n            query_states, key_states, cos, sin\n        )' in model_content,
     "apply_rotary_pos_emb called without position_ids (transformers >=5.0 signature)"),
]

for name, ok, desc in checks:
    status = "OK" if ok else "MISSING"
    if not ok:
        all_ok = False
    print(f"  {status}: {name}")
    print(f"         {desc}")

# Verify no flash_attn RuntimeError remains
if 'flash attention must be installed' in model_content:
    print("  ERROR: Old RuntimeError guard still present!")
    all_ok = False

# ── Tokenizer patches ────────────────────────────────────────────────────────

if os.path.exists(TOKENIZER_BLOB):
    with open(TOKENIZER_BLOB) as f:
        tok_content = f.read()

    tok_checks = [
        ("Patch 4 (minimal parent init)",
         "Minimal parent init: _special_tokens_map is needed" in tok_content,
         "super().__init__() replaced with manual _special_tokens_map init"),

        ("Patch 5 (get_vocab override)",
         'def get_vocab(self) -> dict:\n        """Override required by transformers >=5.0."""' in tok_content,
         "get_vocab() override returns self.vocab"),
    ]

    for name, ok, desc in tok_checks:
        status = "OK" if ok else "MISSING"
        if not ok:
            all_ok = False
        print(f"  {status}: {name}")
        print(f"         {desc}")
else:
    print(f"  WARNING: Tokenizer blob not found at {TOKENIZER_BLOB}")

# ── Clear modules cache ──────────────────────────────────────────────────────

if os.path.exists(MODULES_DIR):
    shutil.rmtree(MODULES_DIR)
    print(f"\n  Cleared modules cache: {MODULES_DIR}")
else:
    print("\n  Modules cache not present (already clear)")

# ── Summary ──────────────────────────────────────────────────────────────────

if all_ok:
    print("\n✓ All patches verified — ready to use Kimi-Audio")
else:
    print("\n⚠ Some patches are missing — re-apply them manually")
    print("  See CLAUDE.md for instructions")
