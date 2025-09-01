#!/bin/bash

# Script to run vLLM server with a model and then execute translation data generation
# Usage: ./generate_translation_data_with_vllm.sh <model_name>

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name>"
    echo "Example: $0 shisa-ai/037-rakuten-2.0-mini-instruct-1.5b-v2new-dpo405b"
    exit 1
fi

MODEL_NAME="$1"
OPENAI_URL="http://localhost:8000/v1"

echo "Starting vLLM server with model: $MODEL_NAME"

# Start vLLM server in background
vllm serve "$MODEL_NAME" \
    --max-model-len 8192 \
    --enforce-eager \
    -O0 &

VLLM_PID=$!

# Wait for server to start
echo "Waiting for vLLM server to start..."
for i in {1..30}; do # 30 * 10s = 5 minutes timeout
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "vLLM server is ready."
        break
    fi
    echo "Waiting for vLLM server... ($i/30)"
    sleep 10
done

# Final check if server is running
if ! curl -s http://localhost:8000/health > /dev/null; then
    echo "vLLM server failed to start after 5 minutes."
    kill $VLLM_PID 2>/dev/null
    wait $VLLM_PID 2>/dev/null
    exit 1
fi

echo "vLLM server started successfully"

# Run translation data generation
echo "Running translation data generation..."
python generate_translation_data.py --base-url "$OPENAI_URL" --test-model-name "$MODEL_NAME"

# Clean up
echo "Shutting down vLLM server..."
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null

echo "Done!"