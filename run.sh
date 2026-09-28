#!/bin/bash
# mini_qwen 一键运行脚本
#
# 用法:
#   bash run.sh smoke     用 mini_data 跑通全流程（约 5 分钟，验证环境是否就绪）
#   bash run.sh pt        预训练
#   bash run.sh sft       微调（依赖 results/pt）
#   bash run.sh dpo       直接偏好优化（依赖 results/sft）
#   bash run.sh eval      交互式对话（默认加载 results/dpo，可追加 eval 脚本参数）
#   bash run.sh all       pt -> sft -> dpo 依次执行
#
# 规模参数可用环境变量覆盖，例如:
#   MAX_SAMPLES=20000 BATCH_SIZE=16 bash run.sh sft
# 完整变量列表见 RUNBOOK.md 第 8.1 节。

set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

# python -u 关闭 stdout 块缓冲，否则重定向到日志时可能几十分钟看不到输出
PY="python -u"
export TQDM_DISABLE="${TQDM_DISABLE:-1}"  # 关进度条，避免日志被 \r 刷屏
export NUM_PROC="${NUM_PROC:-32}"

# 单卡默认等效 batch = BATCH_SIZE * GRAD_ACCUM = 192，与原项目 6 卡设置一致
PT_BS="${BATCH_SIZE:-24}";  PT_GA="${GRAD_ACCUM:-8}"
SFT_BS="${BATCH_SIZE:-24}"; SFT_GA="${GRAD_ACCUM:-8}"
# DPO 每步要跑 4 次前向（策略/参考模型 x chosen/rejected），batch 需减小
DPO_BS="${BATCH_SIZE:-8}";  DPO_GA="${GRAD_ACCUM:-8}"

usage() { sed -n '2,15p' "$0"; }

case "${1:-}" in
  smoke)
    out=/tmp/mini_qwen_smoke
    echo ">>> 冒烟测试：用 mini_data 跑通 pt -> sft -> dpo，产物写到 $out"
    rm -rf "$out"
    DATA_DIR=mini_data MAX_SAMPLES=200 BLOCK_SIZE=256 BATCH_SIZE=4 GRAD_ACCUM=2 NUM_PROC=4 \
      OUTPUT_PATH="$out/pt"  $PY mini_qwen_pt.py
    DATA_DIR=mini_data MAX_SAMPLES=400 MAX_LENGTH=256 BATCH_SIZE=4 GRAD_ACCUM=2 NUM_PROC=4 EPOCHS=1 \
      MODEL_PATH="$out/pt"  OUTPUT_PATH="$out/sft" $PY mini_qwen_sft.py
    DATA_DIR=mini_data MAX_SAMPLES=200 MAX_LENGTH=256 BATCH_SIZE=2 GRAD_ACCUM=2 NUM_PROC=4 EPOCHS=1 \
      MODEL_PATH="$out/sft" OUTPUT_PATH="$out/dpo" $PY mini_qwen_dpo.py
    echo ">>> 冒烟测试通过。模型输出是乱码属正常，只训了几百条样本"
    ;;

  pt)
    BATCH_SIZE="$PT_BS" GRAD_ACCUM="$PT_GA" \
      $PY mini_qwen_pt.py 2>&1 | tee logs/run_pt.log
    ;;

  sft)
    # 全量 50 万条 x 3epoch 单卡约 40 小时，故默认限制为 12000 条
    BATCH_SIZE="$SFT_BS" GRAD_ACCUM="$SFT_GA" MAX_SAMPLES="${MAX_SAMPLES:-12000}" \
      $PY mini_qwen_sft.py 2>&1 | tee logs/run_sft.log
    ;;

  dpo)
    # 全量 59438 条 x 3epoch 单卡约 13 小时，故默认限制为 3000 条
    BATCH_SIZE="$DPO_BS" GRAD_ACCUM="$DPO_GA" MAX_SAMPLES="${MAX_SAMPLES:-3000}" \
      $PY mini_qwen_dpo.py 2>&1 | tee logs/run_dpo.log
    ;;

  eval)
    shift
    if [ "$#" -eq 0 ]; then
      python mini_qwen_eval.py --model_path results/dpo
    else
      python mini_qwen_eval.py "$@"
    fi
    ;;

  all)
    bash "$0" pt && bash "$0" sft && bash "$0" dpo
    ;;

  ""|-h|--help|help) usage ;;
  *) echo "未知参数: $1"; echo; usage; exit 1 ;;
esac
