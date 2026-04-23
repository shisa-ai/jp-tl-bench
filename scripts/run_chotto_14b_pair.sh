#!/usr/bin/env bash
set -euo pipefail

cd /home/aomori/jp-tl-bench

MODELS=(
  "shisa-ai/chotto-14b-20260107-dpo"
  "shisa-ai/chotto-14b-20251007-dpo"
)

SNAP="baseset/zh-ja-chinese-models-v1.0"
TASK="translation_zh_ja_bidirectional_v1"
JUDGE_MODEL="seed-2-0-pro-260328"
JUDGE_PROFILE="cn_judge"
JUDGE_URL="https://ark.ap-southeast.bytepluses.com/api/v3/responses"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
VLLM_GPUS="${VLLM_GPUS:-0,1}"

log_ts() {
  date -u "+%Y-%m-%dT%H:%M:%SZ"
}

load_secret() {
  local name="$1"
  if [ -n "${!name:-}" ]; then
    return
  fi
  local line
  line=$(grep "^export ${name}=" ~/.bashrc 2>/dev/null || true)
  if [ -n "$line" ]; then
    eval "$line"
  fi
}

load_secret HF_TOKEN
load_secret BYTEPLUS_TOKEN
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Missing HF_TOKEN"
  exit 1
fi
if [ -z "${BYTEPLUS_TOKEN:-}" ]; then
  echo "Missing BYTEPLUS_TOKEN"
  exit 1
fi

cleanup_vllm() {
  if [ -n "${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "===== stopping vLLM pid=$VLLM_PID $(log_ts) ====="
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup_vllm EXIT

wait_for_server() {
  for _ in $(seq 1 180); do
    if curl -fsS "http://${HOST}:${PORT}/health" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

for model in "${MODELS[@]}"; do
  safe="${model//\//__}"
  judge_dir="${JUDGE_MODEL}.${JUDGE_PROFILE}"
  echo "===== starting $model $(log_ts) ====="

  cleanup_vllm
  VLLM_PID=""

  CUDA_VISIBLE_DEVICES="$VLLM_GPUS" mamba run -n vllm vllm serve "$model" \
    --served-model-name "$model" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 2 \
    --max-model-len 12288 \
    --gpu-memory-utilization 0.90 \
    --max-num-seqs 32 \
    --dtype bfloat16 \
    --enable-prefix-caching \
    --enforce-eager \
    -O0 > "logs/${safe}.vllm.log" 2>&1 &
  VLLM_PID=$!

  echo "===== waiting for vLLM $model pid=$VLLM_PID $(log_ts) ====="
  if ! wait_for_server; then
    echo "vLLM failed to become healthy for $model"
    tail -200 "logs/${safe}.vllm.log" || true
    exit 1
  fi

  echo "===== generating translations $model $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python generate_translation_data.py \
    --task "$TASK" \
    --base-url "http://${HOST}:${PORT}/v1" \
    --test-model "$model" \
    --api-key-env OPENAI_API_KEY \
    --max-workers 20 \
    --concurrency-limit 20 \
    --max-tokens 8192

  echo "===== generating pairs $model $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python generate_shootout_data.py \
    --task "$TASK" \
    --judge-profile "$JUDGE_PROFILE" \
    --test-model "$model" \
    --judge-model "$JUDGE_MODEL"

  echo "===== judging $model $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python translation_comparer_any_model.py \
    --task "$TASK" \
    --judge-profile "$JUDGE_PROFILE" \
    --base-url "$JUDGE_URL" \
    --judge-model "$JUDGE_MODEL" \
    --test-model "$model" \
    --api-key-env BYTEPLUS_TOKEN \
    --skylark-judge \
    --max-workers 8 \
    --concurrency-limit 8

  echo "===== scoring $model $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python choix_analyzer.py \
    --task "$TASK" \
    --judge-profile "$JUDGE_PROFILE" \
    --test-model "$model" \
    --judge-model "$JUDGE_MODEL"

  echo "===== visualizing $model $(log_ts) ====="
  BASESET_SNAPSHOT_DIR="$SNAP" mamba run -n shisa-jp-tl-bench python score_visualizer.py \
    --baseset-version zh-ja-chinese-models-v1.0 \
    --judge "$JUDGE_MODEL" \
    --task "$TASK" \
    --filter "$model"

  echo "===== completed $model outputs under results/zh-ja-chinese-models-v1.0/${safe}/${judge_dir} $(log_ts) ====="
done

cleanup_vllm
VLLM_PID=""
echo "===== ALL DONE $(log_ts) ====="
