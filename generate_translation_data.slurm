#!/bin/bash
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=209715M
#SBATCH --time=24:00:00
#SBATCH --output=/fsx/ubuntu/_logs/%j-VLLM-multinode.log

# Default values
MODEL="${MODEL:-}"  # Use empty string if MODEL is not set
LOW_CONTEXT="${LOW_CONTEXT:-false}"  # Default to false if not set
OPENAI_URL="${OPENAI_URL:-}"  # API URL for the model
JUDGE_URL="${JUDGE_URL:-http://athenev2/v1}"  # Default judge API URL
JUDGE_MODEL="${JUDGE_MODEL:-Nexusflow/Athene-V2-Chat}"  # Default judge model
CURATOR_DISABLE_CACHE=true
BASE_RESULTS_DIR="/fsx/ubuntu/evals/shisa-tl-bench/logs"

echo "Running on node: $(hostname)"

    # Validate required arguments
if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable is required"
    echo "Usage: MODEL=<model_name> [LOW_CONTEXT=true] sbatch $0"
    exit 1
fi

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

cleanup() {
    local exit_code=$?
    log "Cleaning up (exit code: $exit_code)..."
    
    if [ ! -z "$SERVER_PID" ]; then
        log "Killing VLLM server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
        
    log "Cleanup complete"
    exit $exit_code
}

# Create results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${BASE_RESULTS_DIR}/${SLURM_JOB_ID}-${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Symlink the slurm logs
ln -s "/fsx/ubuntu/_logs/${SLURM_JOB_ID}-VLLM-multinode.log" "${RESULTS_DIR}/slurm.log"

# Activate mamba environment
source /fsx/ubuntu/miniforge3/etc/profile.d/conda.sh
source /fsx/ubuntu/miniforge3/etc/profile.d/mamba.sh
mamba activate /fsx/ubuntu/miniforge3/envs/vllm

# Change to working directory
cd /fsx/ubuntu/evals/shisa-tl-bench

# Calculate cluster size and parallelism
TOTAL_GPUS=$((SLURM_NNODES * $(nvidia-smi -L | wc -l)))
GPUS_PER_NODE=$(nvidia-smi -L | wc -l)
TENSOR_PARALLEL_SIZE=2

log "Cluster configuration:"
log "- Total nodes: $SLURM_NNODES"
log "- Total GPUs: $TOTAL_GPUS"
log "- GPUs per node: $GPUS_PER_NODE"
log "- Tensor parallel size: $TENSOR_PARALLEL_SIZE"

# Function to check if VLLM server is responding
check_server() {
    local host=$1
    local port=$2
    curl -s -f "http://${host}:${port}/v1/models" > /dev/null
    return $?
}
if [ "${SLURM_PROCID}" -eq 0 ]; then
        log "Proceeding with VLLM startup"
    
    # Start vLLM server
    log "Starting vLLM server with tensor parallel size: $TENSOR_PARALLEL_SIZE"
    log "Full vLLM command:"
    log "vllm serve $MODEL --tensor-parallel-size $TENSOR_PARALLEL_SIZE --gpu-memory-utilization 0.95 --num-scheduler-steps 20 --host 0.0.0.0 --port 8000"

    # Create log file for VLLM output
    VLLM_LOG="${RESULTS_DIR}/VLLM_server.log"

    touch "$VLLM_LOG"
    log "VLLM logs will be written to: $VLLM_LOG"
    # Start vLLM with output redirected to log file
    # Build vLLM command with conditional pipeline parallel flag
# Execute the command and capture PID directly
    VLLM_CMD="vllm serve $MODEL --tensor-parallel-size $TENSOR_PARALLEL_SIZE"
    if [ "$LOW_CONTEXT" = "true" ]; then
        VLLM_CMD="$VLLM_CMD --gpu-memory-utilization 0.95 --num-scheduler-steps 20 --max-model-len 8192 --host 0.0.0.0 --port 8000 --trust-remote-code"
    else
        VLLM_CMD="$VLLM_CMD --gpu-memory-utilization 0.95 --num-scheduler-steps 20 --max-model-len 14000 --host 0.0.0.0 --port 8000 --trust-remote-code"
    fi

    log "Full vLLM command:"
    log "$VLLM_CMD"

    # Start VLLM and capture its PID
    $VLLM_CMD > "$VLLM_LOG" 2>&1 & 
    SERVER_PID=$(pgrep -n -f "vllm serve $MODEL")
    log "vLLM server started with PID: $SERVER_PID"
    # Tail the log file in background to show startup progress
    tail -f "$VLLM_LOG" & 
    TAIL_PID=$!
    
    # Wait for server to be ready with timeout and logging
    log "Waiting for VLLM server to become ready..."
    server_attempts=0
    max_server_attempts=720  # 1 hour with 5 second sleep
    
    while true; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            log "ERROR: VLLM server process died. Last 20 lines of log:"
            tail -n 20 "$VLLM_LOG" | sed 's/^/    /'
            kill $TAIL_PID 2>/dev/null  # Stop the log tail
            exit 1
        fi
        
        # Check if server is responding
        if check_server "localhost" 8000; then
            log "VLLM server is ready!"
            break
        fi
        
        # Check log for loading progress
        if tail -n 1 "$VLLM_LOG" | grep -q "Loading safetensors checkpoint shards"; then
            PROGRESS=$(tail -n 1 "$VLLM_LOG" | grep -o "[0-9]\+%" | head -n 1)
            log "Model loading in progress: $PROGRESS"
        fi
        
        server_attempts=$((server_attempts + 1))
        if [ $server_attempts -eq $max_server_attempts ]; then
            log "ERROR: Timeout waiting for VLLM server. Last 20 lines of log:"
            tail -n 20 "$VLLM_LOG" | sed 's/^/    /'
            kill $TAIL_PID 2>/dev/null  # Stop the log tail
            exit 1
        fi
        sleep 5
    done
    
    kill $TAIL_PID 2>/dev/null  # Stop the log tail
    log "VLLM server is ready"
    
    # Now that we're further down, activate shisa-translation-bench
    mamba activate shisa-translation-bench
    log "Starting eval script"
    curl http://localhost:8000/v1/models 2>&1


    if [ "$LOW_CONTEXT" = "true" ]; then
        python generate_translation_data.py --base-url "http://localhost:8000/v1"  --test-model-name $MODEL  --low-context
    else
        python generate_translation_data.py --base-url "http://localhost:8000/v1"  --test-model-name $MODEL 
    fi

    log "Eval script completed successfully."
    # Wait for the server process
    kill $SERVER_PID
    wait $SERVER_PID


else
    # Worker nodes wait for the head node to complete
    wait
fi

log "Job completed"
