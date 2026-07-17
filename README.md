# Audio QA Robustness Evaluation Framework

多模型音频问答鲁棒性评估框架 — 在 MMAU / MMAR 基准上，对多个音频理解模型进行 baseline、静音、噪声、选项打乱、纯文本等扰动条件下的系统性评测。

## 支持的模型

| 模型 | 参数量 | HuggingFace ID | Batch 推理 | 状态 |
|------|--------|----------------|------------|------|
| Qwen2.5-Omni-7B | 7B | `Qwen/Qwen2.5-Omni-7B` | ✅ | ✅ 已验证 |
| Qwen2-Audio-7B-Instruct | 7B | `Qwen/Qwen2-Audio-7B-Instruct` | ✅ | ⬜ 待测试 |
| MOSS-Audio-8B-Instruct | 8B | `OpenMOSS-Team/MOSS-Audio-8B-Instruct` | ✅ | ✅ 已验证 |
| Kimi-Audio-7B-Instruct | 7B | `moonshotai/Kimi-Audio-7B-Instruct` | ✅ | ⬜ 待测试 |

| Mock Model *(测试用)* | - | 本地，无需模型权重 | ✅ | ✅ |

所有真实模型默认使用 `dtype: bfloat16`，`device_map: auto` 自动分配 GPU/CPU/磁盘。

## 支持的基准

| 基准 | 来源 | 规模 | 题型 | 注意 |
|------|------|------|------|------|
| [MMAU](https://mmaubench.github.io/) | ICLR 2025 | 1,000 (test-mini) | 4 选项选择题 | 音频内嵌于 HF 数据集，开箱即用 |
| [MMAR](https://github.com/ddlBoJack/MMAR) | NeurIPS 2025 | 1,000 | 2-6 选项选择题 | 仅含 `audio_path`，音频需另行下载并配置 `audio_root` |

## 扰动条件

| 条件 | 说明 |
|------|------|
| `baseline` | 原始音频，不做任何变换 |
| `silent_audio` | 将音频替换为同长度的静音（全零信号） |
| `noise_audio` | 将音频替换为数据集中其他样本的音频（不相关音频） |
| `shuffled_choices` | 随机打乱选择题选项顺序（正确答案自动 remap） |
| `label_only` | 仅输出选项字母（A/B/C/D），不输出解释文字 |

## 项目结构

```
audioQAagent/
├── configs/                        # YAML 配置
│   ├── base.yaml                   # 全局默认值 (seed, device, batch_size, runtime)
│   ├── models/                     # 模型配置 (5 个真实模型 + 1 个 mock)
│   ├── benchmarks/                 # 基准配置 (mmau, mmar)
│   └── experiments/                # 实验配置 (mmau_full, mmar_full, quick_test)
│
├── src/
│   ├── core/                       # 基础层: types, config, registry
│   ├── data/                       # 数据层: MMAU/MMAR loader, audio_utils
│   ├── models/                     # 模型层: AbstractModel + 6 个 adapter
│   ├── perturbations/             # 扰动层: 6 个条件
│   ├── runner/                     # 执行层: EvaluationTask, Orchestrator, logging
│   └── evaluation/                 # 评估层: metrics, reporter, exporter
│
├── scripts/
│   ├── run_experiment.py           # 运行完整实验
│   ├── run_single.py               # 运行单个 (model, benchmark, perturbation)
│   └── aggregate.py                # 重新聚合已有结果
│
├── tests/fixtures/                 # 测试数据集
├── legacy/infer.py                 # 原始推理脚本 (已归档)
├── third_party/moss_audio_vendor/  # MOSS-Audio 模型代码 (vendored)
└── outputs/                        # 实验结果输出
```

## 快速开始

### 环境配置

项目依赖使用 `audioqa` conda 环境（**必需**）：

```bash
# 激活环境
conda activate audioqa

# 验证依赖
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "import transformers; print('Transformers:', transformers.__version__)"
```

> **已验证的依赖版本：** `transformers==5.13.0`, `torch==2.13.0+cu130`, `torchaudio==2.11.0`, `datasets==5.0.0`, `accelerate==1.14.0`。环境路径：`/home/lt/miniconda3/envs/audioqa/`。**请勿使用 base 环境**，版本不匹配会导致模型加载失败或推理错误。

### 模型与数据下载

首次运行真实模型或基准时，框架会自动从 HuggingFace 下载权重和数据集到 `~/.cache/huggingface/`。

**1️⃣ 设置 HuggingFace Token（必需）**

`Qwen2.5-Omni-7B` 是 gated model，需要登录 HuggingFace 并申请访问权限后才能下载。

```bash
# 方式一：环境变量（推荐，优先级最高）
export HF_TOKEN="hf_your_token_here"

# 方式二：写入配置文件（不推荐，仅供测试）
# 编辑 configs/base.yaml，设置 hf_token: "hf_your_token_here"
```

> 申请 gated model 访问：登录 [HuggingFace](https://huggingface.co) → 进入模型页（如 [Qwen2.5-Omni-7B](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)）→ 点击 "Agree and access repository"。

**2️⃣ 设置镜像（国内网络）**

```bash
# 默认已配置 hf-mirror.com，无需额外操作
# 如需切换：export HF_ENDPOINT="https://hf-mirror.com"
```

**3️⃣ 预下载所有模型（可选）**

首次运行会自动下载，也可提前拉取避免实验中断：

```bash
conda activate audioqa

# 预下载模型权重
python -c "
from huggingface_hub import snapshot_download
snapshot_download('moonshotai/Kimi-Audio-7B-Instruct')
snapshot_download('Qwen/Qwen2.5-Omni-7B')
snapshot_download('Qwen/Qwen2-Audio-7B-Instruct')
snapshot_download('OpenMOSS-Team/MOSS-Audio-8B-Instruct')
snapshot_download('THUDM/glm-4-voice-tokenizer')  # Kimi-Audio 依赖
"

# 预下载数据集
python -c "
from datasets import load_dataset
load_dataset('lmms-lab/mmau', split='test_mini')
load_dataset('BoJack/MMAR', split='test')
"
```

下载后的缓存结构：
```
~/.cache/huggingface/
├── hub/
│   ├── models--moonshotai--Kimi-Audio-7B-Instruct/   # ~35GB
│   ├── models--Qwen--Qwen2.5-Omni-7B/                 # ~15GB
│   ├── models--Qwen--Qwen2-Audio-7B-Instruct/          # ~15GB
│   ├── models--OpenMOSS-Team--MOSS-Audio-8B-Instruct/  # ~16GB
│   └── models--THUDM--glm-4-voice-tokenizer/           # ~1GB
└── datasets/
    ├── lmms-lab___mmau/                                 # ~12GB
    └── BoJack___MMAR/                                   # 音频路径引用
```

### 快速测试（离线，无需模型权重）

```bash
conda activate audioqa

# Mock 模型 + 10 条样本 + batch_size=4
python scripts/run_single.py mock_model mmau baseline --max-samples 10 --batch-size 4

# 完整快速实验（mock × 全部条件）
python scripts/run_experiment.py quick_test
```

### 运行真实实验

```bash
conda activate audioqa

# ===== Step 1: 冒烟测试（推荐先跑 20 条确认正常） =====
python scripts/run_single.py qwen_omni      mmau baseline --max-samples 20 --batch-size 8
python scripts/run_single.py moss_audio_8b  mmau baseline --max-samples 20 --batch-size 8

# ===== Step 2: 运行完整实验 =====
python scripts/run_experiment.py mmau_full
python scripts/run_experiment.py mmar_full

# 只跑指定模型
python scripts/run_experiment.py mmau_full --models qwen_omni
```

### 批处理推理

框架支持样本级批量推理，将多个样本合并为一次 `model.generate()` 调用以减少 GPU 上下文切换开销。

```bash
# CLI 指定 batch_size（默认 4）
python scripts/run_single.py qwen_omni mmau baseline --batch-size 8

# 配置中设置（实验时自动使用）
# configs/base.yaml → runtime.batch_size: 8
```

**按音频长度排序分批**：推理前自动按音频 duration 排序，相近长度样本进入同一 batch，最小化 padding 浪费。日志中可看到排序范围：
```
Sorted 1000 samples by audio duration (range: 2.0s – 34.5s)
```

各模型适配器均实现了 `infer_batch()` 方法，若 batch 推理失败自动回退到逐样本推理。

### 断点续跑

实验默认开启 `resume: true`（见 `configs/base.yaml`），中断后重新运行会自动跳过已完成样本。强制重新运行：

```bash
python scripts/run_single.py qwen_omni mmau baseline --no-resume
```

## 核心设计

### 抽象接口

```python
class AbstractModel(ABC):
    def load(self) -> None
    def infer(audio, question, choices) -> (answer, raw_output)
    def infer_batch(batch) -> [(answer, raw_output), ...]  # 批量推理
    def unload(self) -> None

class AbstractBenchmark(ABC):
    def load(split) -> None
    def __getitem__(idx) -> Sample
    @property category_fields

class Perturbation(ABC):
    def init_context(benchmark) -> None  # 延迟音频池（避免预加载全部音频到内存）
    def apply(sample, rng) -> Sample
```

### Sample 数据流

```
原始数据 (HF dataset)
  → Arrow raw bytes + soundfile 解码（避免 AudioDecoder 内存泄漏）
  → 16 kHz mono WAV tempfile
  → Sample.audio = file path（字符串，轻量）
  → Perturbation.apply()
  → Model.infer() / Model.infer_batch()
  → Prediction + JSONL 增量写入
```

### 执行策略

- **模型外循环** — 加载一次模型跑完所有 benchmark × perturbation，减少 GPU 加载开销
- **长度排序分批** — 推理前按音频 duration 排序，相近长度入同一 batch，降低 padding 浪费
- **批处理 fallback** — batch 推理失败自动降级为逐样本推理，不影响任务完成
- **JSONL 增量保存** — 每条样本立即追加写入，支持断点续跑
- **延迟音频池** — `init_context(benchmark)` 仅存储 benchmark 引用，音频按需读取（避免 1000 条样本同时加载到内存）

### 耗时统计

metadata.json 记录完整耗时分解：

```json
{
  "timing": {
    "total_wall_seconds": 1234.5,
    "model_load_seconds": {"qwen_omni": 45.2},
    "total_model_load_seconds": 45.2,
    "benchmark_load_seconds": {"mmau": 19.1},
    "total_benchmark_load_seconds": 19.1,
    "total_perturbation_seconds": 12.3,
    "total_inference_seconds": 1100.5,
    "per_task": {
      "qwen_omni__mmau__baseline": {
        "duration_seconds": 120.0,
        "samples_per_second": 8.33,
        "perturbation_time_seconds": 0.5,
        "inference_time_seconds": 119.5,
        "accuracy": 0.6543
      }
    }
  }
}
```

tqdm 进度条实时显示 `samples/s`、运行准确率 `acc` 和当前 batch 大小 `bs`。

## 常见问题

### 环境相关问题

**`ModuleNotFoundError: No module named 'torch'`**

原因：未激活 `audioqa` conda 环境。必须在每次新终端中执行 `conda activate audioqa`。

**`ModuleNotFoundError: No module named 'qwen_omni_utils'`**

原因：使用了错误的 Python 环境。确认 `which python` 输出指向 `/home/lt/miniconda3/envs/audioqa/bin/python`。

### 显存不足与 CPU offload

**现象**：GPU 显存占用低但 CPU 和内存占用高，推理速度慢（~3-4s/sample）。

**原因**：7B-8B 模型在 bfloat16 下约 14-19GB，8GB 专用显存无法完全容纳。`device_map: auto` 自动将部分层卸载到 CPU 进行计算，产生 PCIe 传输开销。

**解决方案**：
1. 在对应模型配置中设置 `load_in_4bit: true`（需要 `bitsandbytes`）
2. Windows 原生 + `device_map: "cuda:0"`（利用共享 GPU 内存池，总 24GB）
3. WSL2 下可尝试 `device_map: "sequential"` 以减少磁盘卸载

### MOSS-Audio 推理报错

已在 `third_party/moss_audio_vendor/modeling_moss_audio.py` 中修复 transformers 5.x 兼容性问题。

### HuggingFace 连接问题

镜像和 Token 配置见 [模型与数据下载](#2-设置镜像国内网络) 一节。常见错误：

- `401 Client Error` — Token 未设置或已过期，执行 `export HF_TOKEN="hf_xxx"`
- `403 Forbidden` — 未对 gated model 申请访问权限，去模型页同意协议
- `Connection refused / timeout` — hf-mirror.com 不可达，切换备选镜像：`export HF_ENDPOINT="https://huggingface.co"`
- `Out of disk space` — HF 缓存目录容量不足，清理 `~/.cache/huggingface/hub/` 中不需要的模型

## 扩展

### 添加新模型

```python
# src/models/my_model.py
from src.core.registry import register_model
from src.models.base import AbstractModel

@register_model("my_model")
class MyModel(AbstractModel):
    @property
    def name(self): return "my_model"
    def load(self): ...
    def infer(self, audio, question, choices): ...
    def infer_batch(self, batch): ...  # 可选：覆盖基类默认的逐样本 fallback
    def unload(self): ...
```

### 添加新扰动条件

```python
# src/perturbations/my_perturbation.py
from src.core.registry import register_perturbation
from src.perturbations.base import Perturbation

@register_perturbation("my_perturbation")
class MyPerturbation(Perturbation):
    @property
    def name(self): return "my_perturbation"
    def apply(self, sample, rng, **kwargs): ...
```

### 添加新基准

```python
# src/data/my_benchmark.py
from src.core.registry import register_benchmark
from src.data.base import AbstractBenchmark

@register_benchmark("my_benchmark")
class MyBenchmark(AbstractBenchmark):
    ...
```
