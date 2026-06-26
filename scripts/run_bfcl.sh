#!/usr/bin/env bash
# BFCL V4 同机评测：对一个本地模型目录跑 generate + evaluate，输出八类分数。
#
# 用官方 `bfcl-eval` 包（Apache-2.0），后端 vLLM 本地推理。base / merged / W4A16
# 三个模型各跑一次、各给独立 result-dir / score-dir，保证 before/after 同机同 vLLM 公平对比。
#
# 关键约定:
#   - 模型名必须带 `Qwen/` 前缀 → 命中 QwenFCHandler(本地 vLLM, FC 模式)。
#     不带前缀(qwen3-14b-FC) 会被路由到云 DashScope API → 401。靠 --local-model-path 区分实际权重。
#   - 必须设 OPENAI_API_KEY=dummy：QwenFCHandler 构造时会 new 一个 OpenAI client(本地 vllm 走
#     OpenAI 兼容口)，缺 key 直接崩；本地推理不真用这个 key。
#   - 八类 = AST(simple_python/multiple/parallel/parallel_multiple) + live(live_simple/live_multiple/
#     live_parallel) + irrelevance。看分类明细，别看 BFCL overall(把没跑的 multi_turn/web_search 当 0)。
#
# 跑法:
#   scripts/run_bfcl.sh /path/to/Qwen3-14B            base
#   scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged  merged
#   scripts/run_bfcl.sh /path/to/Qwen3-14B-fc-merged-W4A16 w4a16
set -euo pipefail

MODEL_PATH="${1:?usage: run_bfcl.sh <local-model-path> <tag>}"
TAG="${2:?usage: run_bfcl.sh <local-model-path> <tag>}"

CATS="simple_python,multiple,parallel,parallel_multiple,irrelevance,live_simple,live_multiple,live_parallel"
RESULT_DIR="results_${TAG}"
SCORE_DIR="scores_${TAG}"

export OPENAI_API_KEY=dummy

echo ">>> [generate] ${TAG}  <-  ${MODEL_PATH}"
bfcl generate \
  --model Qwen/Qwen3-14B-FC \
  --local-model-path "${MODEL_PATH}" \
  --backend vllm --num-gpus 1 --gpu-memory-utilization 0.85 \
  --test-category "${CATS}" \
  --result-dir "${RESULT_DIR}"

echo ">>> [evaluate] ${TAG}"
bfcl evaluate \
  --model Qwen/Qwen3-14B-FC \
  --test-category "${CATS}" \
  --result-dir "${RESULT_DIR}" \
  --score-dir "${SCORE_DIR}"

echo ">>> done: per-category scores in ${SCORE_DIR}/Qwen_Qwen3-14B-FC/"
