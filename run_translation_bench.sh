#!/bin/bash

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
ULTRA_LOW_CONTEXT="${ULTRA_LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-}"  # Judge API URL (no default)
JUDGE_MODEL="${JUDGE_MODEL:-Nexusflow/Athene-V2-Chat}"  # Default judge model
CURATOR_DISABLE_CACHE=true

# Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is missing"
    echo "Usage: MODEL=<model_name> [OPENAI_URL=<api_url>] [JUDGE_URL=<judge_url>] [LOW_CONTEXT=true] [ULTRA_LOW_CONTEXT=true] [JUDGE_MODEL=model_name] [CURATOR_CACHE_DIR=path] ./$0"
    echo "Examples:"
    echo "  MODEL=mistral OPENAI_URL=http://localhost:8000/v1 ./$0"
    echo "  MODEL=/path/to/model/checkpoint ./$0"
    exit 1
fi

# Remove trailing slash from MODEL if present
MODEL="${MODEL%/}"

# Process model name for display and reference
MODEL_FULL_PATH="$MODEL"
if [ -d "$MODEL" ]; then
    # It's a local path that exists
    BASENAME=$(basename "$MODEL")
    if [[ "$BASENAME" == checkpoint* ]]; then
        # If it's a checkpoint dir, use containing_folder.checkpoint_dir as name
        CONTAINING_FOLDER=$(basename "$(dirname "$MODEL")")
        MODEL_NAME="${CONTAINING_FOLDER}.${BASENAME}"
    else
        # Otherwise, use the basename
        MODEL_NAME="$BASENAME"
    fi
else
    # If not an existing path, assume it's an HF model name and use it directly
    MODEL_NAME="$MODEL"
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

# Log the model information
log "Using model path: $MODEL_FULL_PATH"
log "Using model name: $MODEL_NAME"

# Initialize and activate conda/mamba
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate shisa-translation-bench

# Change to working directory
log "Starting eval script"

# Define context flags based on settings
if [ "$ULTRA_LOW_CONTEXT" = "true" ]; then
    CONTEXT_FLAG="--ultra-low-context"
elif [ "$LOW_CONTEXT" = "true" ]; then
    CONTEXT_FLAG="--low-context"
else
    CONTEXT_FLAG=""
fi

# Check if OPENAI_URL is specified
if [ -n "$OPENAI_URL" ]; then
    # Use the provided OPENAI_URL
    log "Using provided OPENAI_URL: $OPENAI_URL"
    CURATOR_DISABLE_CACHE=true python generate_translation_data.py --base-url $OPENAI_URL --test-model-name $MODEL $CONTEXT_FLAG
else
    # Run the model with vLLM via slurm
    log "No OPENAI_URL provided, running model with vLLM via slurm"
    
    
    # Check if context flags were explicitly set
    if [ "$ULTRA_LOW_CONTEXT" = "true" ]; then
        log "Using ultra-low context (4096) as explicitly requested"
        MODEL_PATH="$MODEL_FULL_PATH" MODEL_NAME="$MODEL_NAME" ULTRA_LOW_CONTEXT="true" srun --priority=1000000 generate_translation_data.slurm --ultra-low-context
    elif [ "$LOW_CONTEXT" = "true" ]; then
        log "Using low context as explicitly requested"
        MODEL_PATH="$MODEL_FULL_PATH" MODEL_NAME="$MODEL_NAME" LOW_CONTEXT="true" srun --priority=1000000 generate_translation_data.slurm --low-context
    # Otherwise, determine context based on model name
    elif [[ " ${ultra_low_context_models[@]} " =~ " ${MODEL_NAME} " ]]; then
        log "Using ultra-low context (4096) based on model type"
        MODEL_PATH="$MODEL_FULL_PATH" MODEL_NAME="$MODEL_NAME" ULTRA_LOW_CONTEXT="true" srun --priority=1000000 generate_translation_data.slurm --ultra-low-context
    elif [[ " ${long_context_models[@]} " =~ " ${MODEL_NAME} " ]]; then
        log "Using standard context settings based on model type"
        MODEL_PATH="$MODEL_FULL_PATH" MODEL_NAME="$MODEL_NAME" srun --priority=1000000 generate_translation_data.slurm
    else
        log "Using standard context settings with low-context flag"
        MODEL_PATH="$MODEL_FULL_PATH" MODEL_NAME="$MODEL_NAME" LOW_CONTEXT="true" srun --priority=1000000 generate_translation_data.slurm --low-context
    fi
fi

log "Successfully generated conversation data. Generating shootout data..."
CURATOR_DISABLE_CACHE=true python generate_shootout_data.py --target-model "$MODEL_NAME"
log "Successfully generated shootout data. Evaluating results with Athene..."

# Check if JUDGE_URL is specified
if [ -n "$JUDGE_URL" ]; then
    # Use the provided JUDGE_URL
    log "Using provided JUDGE_URL: $JUDGE_URL"
    CURATOR_DISABLE_CACHE=true python translation_comparer_any_model.py --base-url "$JUDGE_URL" --judge-model-name "$JUDGE_MODEL" --test-model-name "$MODEL_NAME"
else
    # Use commercial model flag when JUDGE_URL is not specified
    log "No JUDGE_URL provided, using commercial model flag"
    CURATOR_DISABLE_CACHE=true python translation_comparer_any_model.py --use-commercial-model --judge-model-name "$JUDGE_MODEL" --test-model-name "$MODEL_NAME"
fi

log "Successfully evaluated results. Running Bradley-Terry comparision..."
python choix_analyzer.py --target-model "$MODEL_NAME" --judge-model "$JUDGE_MODEL"
log "All done! Scores saved to scores/scores.jsonl"
