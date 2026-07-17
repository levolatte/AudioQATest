# CLAUDE.md — Audio QA Robustness Evaluation

## Project overview

Multi-model robustness evaluation framework for audio question-answering benchmarks.
Evaluates real and mock models on MMAU/MAR datasets under various perturbations
(noise, silence, shuffled choices, label-only prompts).

Structure:
```
src/core/       — types, config loading, registry (decorator-based plugin system)
src/data/       — benchmark loaders (MMAU, MMAR), audio utilities
src/models/     — model adapters (Qwen Omni, Qwen2-Audio, Kimi-Audio, MOSS-Audio, Mock)
src/perturbations/ — perturbation strategies (baseline, noise, silent, shuffled, label_only)
src/runner/     — orchestrator (grid engine), evaluation task (single run), logging
src/evaluation/ — metrics, CSV/Markdown report export
legacy/         — reference implementation (infer.py): the gold standard for model behavior
configs/        — YAML configs for experiments, models, benchmarks
scripts/        — entry points: run_experiment.py, run_single.py, aggregate.py
```

## ⚠️ Critical warnings

### ALWAYS use the `audioqa` conda environment

This project **requires** the `audioqa` conda environment. All scripts, tests, and
experiments MUST be run with it:

```bash
conda activate audioqa
python scripts/run_single.py ...    # ✅ correct
python scripts/run_single.py ...    # ❌ wrong if audioqa not active
```

The environment lives at `/home/lt/miniconda3/envs/audioqa/` and contains:
- PyTorch 2.13+cu130, transformers 5.13, datasets 5.0, accelerate 1.14
- `qwen_omni_utils`, `moss_audio_vendor` (vendored under `third_party/`)
- All audio I/O libraries (soundfile, librosa, torchaudio)

**Never** run experiments from the base conda environment or any other env — missing
dependencies or wrong versions will cause cryptic failures.

### HuggingFace AudioDecoder memory leak (DO NOT USE)

The `datasets` library's `AudioDecoder` feature (torchcodec) has a **C++-level memory
leak**. Every access to `ds[i]["audio"]` or `ds[i].audio` through the decode path
caches decoded audio in native memory that Python's GC cannot reclaim. For 1000
samples, RSS grows by 1-2 GB+, triggering the WSL OOM killer (exit code 137).

**The fix (ALWAYS use this pattern for HF audio):**
1. Load dataset: `ds = hf_load_dataset(...)`
2. Strip audio column for metadata: `meta_ds = ds.remove_columns(["audio"])`
3. Get raw Arrow handle: `audio_arrow = ds.data.column("audio")`
4. Read raw bytes: `raw_bytes = audio_arrow[i].as_py()["bytes"]`
5. Decode with `soundfile`: `sf.read(io.BytesIO(raw_bytes))`
6. Resample to 16 kHz with `torchaudio.functional.resample()`
7. Save to temp WAV, store file path string in Sample
8. **Drop ALL HF references**: `del ds, meta_ds, arrow_table, audio_arrow`
9. **Never** store HF dataset references for lazy loading — they pin the Arrow table

See `src/data/mmau.py:_load_from_hf()` and `_extract_audio_sample()` for the
reference implementation.

### WSL / limited memory environment

- This project runs on WSL2 with constrained RAM.
- **Never run full 1000-sample experiments with real models without explicit confirmation.**
- Before full runs, verify with `--max-samples 20` + mock_model first.
- After each batch of real model inference, `torch.cuda.empty_cache()` is called
  by the orchestrator. If adding new code paths, ensure GPU cleanup.

### Legacy code is the reference

`legacy/infer.py` is the **authoritative reference** for how models should behave.
It loads a local JSONL file, iterates samples in a simple `for item in tqdm(data)`
loop, and audio is always a **file path string** — the model reads files itself.
No HF datasets, no AudioDecoder, no memory issues. When debugging model behavior,
**always compare against legacy first.**

## Data flow

```
Config (YAML)
  → Orchestrator.run()
    → [outer loop] Model.load()
    → [mid loop]   Benchmark.load()  → _load_from_hf() or _load_from_file()
                     → list[Sample] stored in self._data
                     → Sample.audio is a **file path string** (temp WAV)
    → [inner loop] Perturbation.apply(sample)
                     → returns transformed Sample (may swap audio path)
    → EvaluationTask.run()
      → for i in range(len(benchmark)):
          sample = benchmark[i]            # simple list lookup (no HF)
          transformed = perturbation.apply(sample)
          chosen, raw = model.infer(transformed.audio, ...)
          write prediction to JSONL
      → ResultSet → CSV / Markdown report
    → Model.unload()
```

### What Sample.audio is (by stage):

| Stage | Type | Notes |
|-------|------|-------|
| After Benchmark.load | `str` (temp WAV path) or `None` | Always 16 kHz mono WAV |
| After noise_audio | `str` (path to other sample) | Lazy access via benchmark[idx].audio |
| After silent_audio | `str` (new temp WAV) | Silence matching original duration |
| After baseline/shuffled/label_only | unchanged `str` | |

All models handle both `str` paths and `torch.Tensor` inputs via
`_tensor_to_tempfile()` in `src/models/base.py`.

## Model reference notes

Each model adapter must match the behavior of `legacy/infer.py`. Key patterns
from legacy:

### Shared prompt & answer extraction (`src/models/utils.py`)

Extracted directly from `legacy/infer.py`:

- `normalize_text()` — `re.sub(r"\s+", " ", str(x).strip().lower())`
- `build_question_prompt()` — wraps question + choices. Two modes:
  - **Standard**: `"Please choose the answer from the following options: [...]. Output the final answer in <answer> </answer>."`
  - **label_only**: `"Output ONLY the letter of the correct choice (e.g. A, B, C, or D). Do NOT include any explanation or the option text.\n\nAnswer:"`
- `clean_answer()` — 7-level fallback (same as legacy):
  1. Parse `<answer>...</answer>` tags
  2. Strip special tokens (`<|im_end|>`, `<|endoftext|>`)
  3. Exact match against option text
  4. Letter match (A/B/C/D)
  5. Regex letter extraction
  6. Option text substring in output
  7. Output substring in option text
  8. Default to `choices[0]`

### Qwen2.5-Omni-7B (`src/models/qwen_omni.py`)

- **Reference**: `legacy/infer.py` — this model adapter IS a refactored version of legacy.
- Uses `Qwen2_5OmniForConditionalGeneration` + `AutoProcessor`
- **CRITICAL**: Uses `qwen_omni_utils.process_mm_info(messages, ...)` to extract audio
  from messages. This function reads audio files from disk — it **requires file paths**,
  not tensors. If given a tensor, write it to a temp file first.
- Message format (OpenAI-style with system prompt):
  ```python
  messages = [
      {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
      {"role": "user", "content": [
          {"type": "audio", "audio": audio_path},
          {"type": "text", "text": prompt},
      ]},
  ]
  ```
- `supports_text_only = True` — can do pure text inference (no audio in user content).
- SYSTEM_PROMPT must match legacy exactly.
- When debugging Qwen Omni issues, run `legacy/infer.py` first to verify the
  model/prompt/audio path combination works, then compare with the adapter.

### Qwen2-Audio-7B (`src/models/qwen2_audio.py`)

- Uses `Qwen2AudioForConditionalGeneration` + `AutoProcessor`
- **Different from legacy**: Qwen2-Audio was NOT in legacy code. No legacy reference exists.
  Behavior must be verified independently against the model's HF documentation.
- Message format: single user message, audio via `audio_url` key:
  ```python
  messages = [{"role": "user", "content": [
      {"type": "audio", "audio_url": audio_path},
      {"type": "text", "text": prompt},
  ]}]
  ```
- Processor call: `processor(text=text, audios=[audio_path], ...)`
- `supports_text_only = False` — audio content block is always included.

### Kimi-Audio-7B (`src/models/kimi_audio.py`)

- Uses `AutoModelForCausalLM` + `AutoProcessor` (custom modeling via `trust_remote_code=True`)
- Repository: `moonshotai/Kimi-Audio-7B-Instruct`
- Message format: similar to Qwen Omni but simpler:
  ```python
  messages = [{"role": "user", "content": [
      {"type": "audio", "audio": audio_path},
      {"type": "text", "text": prompt},
  ]}]
  ```
- For text-only (no audio): `messages = [{"role": "user", "content": prompt}]` (plain string)
- Processor: can accept `audio=audio_path` or `text` only.
- `supports_text_only = False`

### MOSS-Audio (`src/models/moss_audio.py`)

- Uses **vendored code** from `third_party/moss_audio_vendor/` (added to sys.path at load time)
- One alias registered: `moss_audio_8b`
- Custom `MossAudioProcessor` with `mel_sr` config for audio loading
- Custom `load_audio()` function: `moss_audio_vendor.audio_io.load_audio(path, sample_rate=...)`
- Processor call includes `audio_token_id` and `audio_data` dtype casting:
  ```python
  inputs["audio_data"] = inputs["audio_data"].to(self._model.dtype)
  inputs["audio_token_id"] = self._processor.audio_token_id
  ```
- `supports_text_only = False`

### Mock model (`src/models/mock_model.py`)

- Pipeline testing model — always returns a fixed answer (first choice or random).
- No weights loaded; no GPU needed.
- Use for smoke-testing data loading and perturbation pipelines.
- Always test with mock_model before running real models.

## Common tasks

### Adding a new model

1. Create `src/models/<name>.py`
2. Subclass `AbstractModel` (`src/models/base.py`)
3. Implement `load()`, `infer()`, `unload()`, `name`, `supports_text_only`
4. Use `build_question_prompt()` and `clean_answer()` from `src/models/utils.py`
5. Register with `@register_model("name")`
6. Add config in `configs/models/<name>.yaml`
7. Import in `src/runner/orchestrator.py` to trigger registration
8. Check against legacy `infer.py` if it's a Qwen-family model

### Adding a new benchmark

1. Create `src/data/<name>.py`
2. Subclass `AbstractBenchmark` (`src/data/base.py`)
3. Implement `load()`, `__len__()`, `__getitem__()`, `name`, `category_fields`
4. **If loading from HF with audio**: follow the Arrow-raw-bytes pattern in `mmau.py:_load_from_hf()`. NEVER use AudioDecoder.
5. **If loading from local files**: store file path strings in Sample, same as legacy.
6. Register with `@register_benchmark("name")`
7. Add config in `configs/benchmarks/<name>.yaml`

### Running experiments

```bash
# Single task (one model, one benchmark, one perturbation):
python scripts/run_single.py <model> <benchmark> <perturbation> [--max-samples N] [--no-resume]

# Full experiment grid:
python scripts/run_experiment.py <experiment_name>

# Quick smoke test (mock model, 20 samples):
python scripts/run_single.py mock_model mmau baseline --max-samples 20 --no-resume
```

### Debugging a crash

1. Check if it's AudioDecoder-related: look for `torchcodec` or `AudioDecoder` in traceback.
   If yes, the code is using the forbidden decode path — switch to Arrow raw bytes.
2. Check exit code: 137 = Linux OOM killer. Memory exhaustion, not a Python error.
3. Run `legacy/infer.py` to verify the model works in isolation.
4. Test with `mock_model` first to isolate data loading from model inference.
5. Test with `--max-samples 20` before running full 1000.
6. Monitor with `htop` or `free -m` to watch RSS growth.

## Key conventions

- **README sync**: Any feature, API, CLI flag, or config change that affects how users
  run or extend the framework MUST update `README.md` in the same commit. This includes
  new CLI arguments, new model/perturbation support, changed defaults (batch_size,
  max_new_tokens), new config fields, or environment requirement changes.
- **Registry pattern**: `@register_model`, `@register_benchmark`, `@register_perturbation`
  decorators auto-register classes. Orchestrator imports trigger registration via
  `import src.data.mmau` etc. in `orchestrator.py`.
- **Config**: Pydantic models in `src/core/types.py`, YAML files in `configs/`. Config
  objects flow from orchestrator → model/benchmark constructors. Adding a new field
  to a Pydantic model requires updating the corresponding YAML configs.
- **Resume**: JSONL per-task prediction files. `EvaluationTask.run()` reads existing
  lines, skips completed `sample_id`s. Set `runtime.resume: true` in config.
- **Temp files**: Audio extracted from HF datasets goes to `tempfile.mkdtemp(prefix="mmau_audio_")`.
  Cleaned up via `atexit.register(self._cleanup_temp_files)`. Model adapters that
  create temp files for tensor→path conversion clean up immediately after inference.
- **Perturbation init_context**: Accepts a `benchmark` object (not a list of samples).
  Access audio lazily via `benchmark[idx].audio` to avoid loading all audio into RAM.
