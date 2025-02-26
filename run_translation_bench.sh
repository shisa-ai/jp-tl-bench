#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
ULTRA_LOW_CONTEXT="${ULTRA_LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-http://athenev2/v1}"  # Default judge API URL
JUDGE_MODEL="${JUDGE_MODEL:-Nexusflow/Athene-V2-Chat}"  # Default judge model
CURATOR_DISABLE_CACHE=true

# Validate required arguments
if [ -z "$MODEL" ] || [ -z "$OPENAI_URL" ]; then
    echo "Error: Required environment variables are missing"
    echo "Usage: MODEL=<model_name> OPENAI_URL=<api_url> [JUDGE_URL=<judge_url>] [LOW_CONTEXT=true] [ULTRA_LOW_CONTEXT=true] [JUDGE_MODEL=model_name] ./$0"
    echo "Example:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 ./$0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Initialize and activate conda/mamba
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate shisa-translation-bench

# Change to working directory
log "Starting eval script"

if [ "$ULTRA_LOW_CONTEXT" = "true" ]; then
    CURATOR_DISABLE_CACHE=true python generate_translation_data.py --base-url $OPENAI_URL  --test-model-name $MODEL  --ultra-low-context
elif [ "$LOW_CONTEXT" = "true" ]; then
    CURATOR_DISABLE_CACHE=true python generate_translation_data.py --base-url $OPENAI_URL  --test-model-name $MODEL  --low-context
else
    CURATOR_DISABLE_CACHE=true python generate_translation_data.py --base-url $OPENAI_URL  --test-model-name $MODEL 
fi

log "Successfully generated conversation data. Generating shootout data..."
CURATOR_DISABLE_CACHE=true python generate_shootout_data.py --target-model "$MODEL"
log "Successfully generated shootout data. Evaluating results with Athene..."
CURATOR_DISABLE_CACHE=true python translation_comparer_any_model.py --base-url "$JUDGE_URL" --judge-model-name "$JUDGE_MODEL" --test-model-name "$MODEL"
log "Successfully evaluated results. Running Bradley-Terry comparision..."
python choix_analyzer.py --target-model "$MODEL" --judge-model "$JUDGE_MODEL"
log "All done! Scores saved to scores/scores.jsonl"
