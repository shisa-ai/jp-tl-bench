#!/bin/bash

set -e
set -o pipefail

MODEL="${MODEL:-}"
TASK_CONFIG="${TASK_CONFIG:-}"
JUDGE_PROFILE="${JUDGE_PROFILE:-default}"
LOW_CONTEXT="${LOW_CONTEXT:-false}"
ULTRA_LOW_CONTEXT="${ULTRA_LOW_CONTEXT:-false}"
OPENAI_URL="${OPENAI_URL:-}"
JUDGE_URL="${JUDGE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"
MODEL_API_KEY_ENV="${MODEL_API_KEY_ENV:-OPENAI_API_KEY}"
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-GEMINI_API_KEY}"
BASESET_SNAPSHOT_DIR="${BASESET_SNAPSHOT_DIR:-baseset/v1.0}"
BENCH_ENV_NAME="${BENCH_ENV_NAME:-shisa-jp-tl-bench}"
MAX_TOKENS="${MAX_TOKENS:-}"

SAFE_MODEL_NAME="${MODEL//\//__}"
SAFE_JUDGE_NAME="${JUDGE_MODEL//\//__}"
if [ "$JUDGE_PROFILE" = "default" ]; then
    JUDGE_RESULT_DIR="$SAFE_JUDGE_NAME"
else
    SAFE_JUDGE_PROFILE="${JUDGE_PROFILE//\//__}"
    JUDGE_RESULT_DIR="$SAFE_JUDGE_NAME.$SAFE_JUDGE_PROFILE"
fi

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ] || [ -z "$TASK_CONFIG" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: TASK_CONFIG=<task_name_or_yaml> MODEL=<model_name> OPENAI_URL=<api_url> [JUDGE_URL=<judge_url>] [JUDGE_PROFILE=<judge_profile>] [LOW_CONTEXT=true] [ULTRA_LOW_CONTEXT=true] [JUDGE_MODEL=model_name] [MODEL_API_KEY_ENV=MY_KEY_VAR] [JUDGE_API_KEY_ENV=MY_JUDGE_KEY_VAR] [BENCH_ENV_NAME=shisa-jp-tl-bench] ./$0"
    echo "Example:"
    echo "  TASK_CONFIG=translation_ja_en_bidirectional_v1 MODEL=mistral OPENAI_URL=http://localhost:8000/v1 ./$0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Initialize and activate conda/mamba (portable)
if command -v mamba &> /dev/null; then
    CONDA_COMMAND="mamba"
elif command -v conda &> /dev/null; then
    CONDA_COMMAND="conda"
else
    echo "Error: Neither mamba nor conda found. Please install one of them."
    exit 1
fi

log "Starting eval script"
log "Task config: $TASK_CONFIG"
log "Judge profile: $JUDGE_PROFILE"
log "Environment: $BENCH_ENV_NAME (via $CONDA_COMMAND run -n)"

# Build arguments for translation generation
GEN_ARGS=(--task "$TASK_CONFIG" --base-url "$OPENAI_URL" --test-model "$MODEL" --api-key-env "$MODEL_API_KEY_ENV")

if [ -n "$MAX_TOKENS" ]; then
    GEN_ARGS+=(--max-tokens "$MAX_TOKENS")
fi

if [ "$ULTRA_LOW_CONTEXT" = "true" ]; then
    GEN_ARGS+=(--ultra-low-context)
elif [ "$LOW_CONTEXT" = "true" ]; then
    GEN_ARGS+=(--low-context)
fi

"$CONDA_COMMAND" run -n "$BENCH_ENV_NAME" python generate_translation_data.py "${GEN_ARGS[@]}"

log "Successfully generated task outputs. Generating comparison pairs..."
"$CONDA_COMMAND" run -n "$BENCH_ENV_NAME" python generate_shootout_data.py --task "$TASK_CONFIG" --judge-profile "$JUDGE_PROFILE" --test-model "$MODEL" --judge-model "$JUDGE_MODEL"
log "Successfully generated comparison pairs. Judging pairs..."
"$CONDA_COMMAND" run -n "$BENCH_ENV_NAME" python translation_comparer_any_model.py --task "$TASK_CONFIG" --judge-profile "$JUDGE_PROFILE" --base-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" --test-model "$MODEL" --api-key-env "$JUDGE_API_KEY_ENV"
log "Successfully judged pairs. Running Bradley-Terry scoring..."
"$CONDA_COMMAND" run -n "$BENCH_ENV_NAME" python choix_analyzer.py --task "$TASK_CONFIG" --judge-profile "$JUDGE_PROFILE" --test-model "$MODEL" --judge-model "$JUDGE_MODEL"
log "All done! Scores saved under results/$(basename ${BASESET_SNAPSHOT_DIR:-baseset/v1.0})/$SAFE_MODEL_NAME/$JUDGE_RESULT_DIR/"
