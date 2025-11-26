#!/bin/bash

set -e
set -o pipefail

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
ULTRA_LOW_CONTEXT="${ULTRA_LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-https://generativelanguage.googleapis.com/v1beta/openai/}"  # Default judge API URL
JUDGE_MODEL="${JUDGE_MODEL:-gemini-2.5-flash}"  # Default judge model
MODEL_API_KEY_ENV="${MODEL_API_KEY_ENV:-OPENAI_API_KEY}" # Default env var for test model API key
JUDGE_API_KEY_ENV="${JUDGE_API_KEY_ENV:-GEMINI_API_KEY}" # Default env var for judge model API key
BASESET_SNAPSHOT_DIR="${BASESET_SNAPSHOT_DIR:-baseset/v1.0}"  # Default anchor snapshot used for comparisons
# Optional override for completion tokens; forwarded to generate_translation_data.py --max-tokens when set.
MAX_TOKENS="${MAX_TOKENS:-}"

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: MODEL=<model_name> OPENAI_URL=<api_url> [JUDGE_URL=<judge_url>] [LOW_CONTEXT=true] [ULTRA_LOW_CONTEXT=true] [JUDGE_MODEL=model_name] [MODEL_API_KEY_ENV=MY_KEY_VAR] [JUDGE_API_KEY_ENV=MY_JUDGE_KEY_VAR] ./$0"
    echo "Example:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 ./$0"
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

. "$(dirname $(which $CONDA_COMMAND))/../etc/profile.d/conda.sh"
. "$(dirname $(which $CONDA_COMMAND))/../etc/profile.d/mamba.sh" &> /dev/null || true

$CONDA_COMMAND activate shisa-jp-tl-bench

# Change to working directory
log "Starting eval script"

# Build arguments for translation generation
GEN_ARGS=(--base-url "$OPENAI_URL" --test-model "$MODEL" --api-key-env "$MODEL_API_KEY_ENV")

if [ -n "$MAX_TOKENS" ]; then
    GEN_ARGS+=(--max-tokens "$MAX_TOKENS")
fi

if [ "$ULTRA_LOW_CONTEXT" = "true" ]; then
    GEN_ARGS+=(--ultra-low-context)
elif [ "$LOW_CONTEXT" = "true" ]; then
    GEN_ARGS+=(--low-context)
fi

python generate_translation_data.py "${GEN_ARGS[@]}"

log "Successfully generated conversation data. Generating shootout data..."
python generate_shootout_data.py --test-model "$MODEL" --judge-model "$JUDGE_MODEL"
log "Successfully generated shootout data. Evaluating results..."
python translation_comparer_any_model.py --base-url "$JUDGE_URL" --judge-model "$JUDGE_MODEL" --test-model "$MODEL" --api-key-env "$JUDGE_API_KEY_ENV"
log "Successfully evaluated results. Running Bradley-Terry comparision..."
python choix_analyzer.py --test-model "$MODEL" --judge-model "$JUDGE_MODEL"
log "All done! Scores saved under results/$(basename ${BASESET_SNAPSHOT_DIR:-baseset/v1.0})/${MODEL//\//__}/${JUDGE_MODEL//\//__}/"
