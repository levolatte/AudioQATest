#!/bin/bash
#SBATCH -J test_inference
#SBATCH -p debug
#SBATCH --nodes=1
#SBATCH --nodelist=GPU2
#SBATCH --gres=gpu:1
#SBATCH --time=0-23:30:00
#SBATCH --output=job/test_inference_%j.log
#SBATCH --error=job/test_inference_%j.log
#SBATCH --qos=normal

# ============================================================
# 四模型推理验证脚本
# 验证 qwen_omni / qwen2_audio / kimi_audio / moss_audio_8b
# 在 MMAU test_mini 上各跑 5 条 baseline 推理，确认模型能正常加载和输出
# ============================================================

# --- 环境准备 ---
source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate audioqa

# 清除旧代理，设置镜像
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY
export HF_ENDPOINT=https://hf-mirror.com
export PYTHONUNBUFFERED=1

# 切到项目根目录
cd ~/projects/audioQAagent 2>/dev/null || cd "$(dirname "$0")/.."

echo "============================================================"
echo "  AudioQA 四模型推理验证"
echo "  开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  节点: $(hostname)"
echo "  CUDA: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

# --- 要测试的模型列表 ---
MODELS=(
    "qwen_omni"
    "qwen2_audio"
    "kimi_audio"
    "moss_audio_8b"
)

BENCHMARK="mmau"
PERTURBATION="baseline"
MAX_SAMPLES=5
OUTPUT_DIR="outputs/test_inference"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS
declare -A DURATIONS

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "  >>> 测试模型: ${MODEL}"
    echo "  >>> 开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "------------------------------------------------------------"

    START_TS=$(date +%s)

    python -u scripts/run_single.py \
        "${MODEL}" \
        "${BENCHMARK}" \
        "${PERTURBATION}" \
        --max-samples "${MAX_SAMPLES}" \
        --batch-size 1 \
        --output-dir "${OUTPUT_DIR}/${MODEL}" \
        --no-resume

    EXIT_CODE=$?
    END_TS=$(date +%s)
    ELAPSED=$((END_TS - START_TS))

    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "  [PASS] ${MODEL} — 耗时 ${ELAPSED}s"
        RESULTS["${MODEL}"]="PASS"
        DURATIONS["${MODEL}"]="${ELAPSED}s"
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        echo "  [FAIL] ${MODEL} — 退出码 ${EXIT_CODE}，耗时 ${ELAPSED}s"
        RESULTS["${MODEL}"]="FAIL (exit=${EXIT_CODE})"
        DURATIONS["${MODEL}"]="${ELAPSED}s"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # 释放 GPU 缓存，为下一个模型腾空间
    python -u -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
done

# --- 汇总 ---
echo ""
echo "============================================================"
echo "  推理验证完成"
echo "  结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "  结果汇总:"
echo "  ----------------------------------------"
printf "  %-20s | %-10s | %-10s\n" "MODEL" "RESULT" "DURATION"
echo "  ----------------------------------------"
for MODEL in "${MODELS[@]}"; do
    printf "  %-20s | %-10s | %-10s\n" "${MODEL}" "${RESULTS[${MODEL}]}" "${DURATIONS[${MODEL}]}"
done
echo "  ----------------------------------------"
echo "  PASS: ${PASS_COUNT} / ${#MODELS[@]}"
echo "  FAIL: ${FAIL_COUNT} / ${#MODELS[@]}"
echo ""
echo "  日志目录: ${OUTPUT_DIR}/"
echo "============================================================"

# 全部通过才返回 0，否则返回失败数
exit ${FAIL_COUNT}
