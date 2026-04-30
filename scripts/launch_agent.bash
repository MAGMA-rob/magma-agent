#!/bin/bash
set -e

# Default values
GPU_ID=""
PORT=""
COMMANDER_ID="none"
MEMORIZER_ID="none"
OUTPUT_STYLE="qwen_format"

while getopts "g:p:c:m:o:" opt; do
  case $opt in
    g) GPU_ID=$OPTARG ;;
    p) PORT=$OPTARG ;;
    c) COMMANDER_ID=$OPTARG ;;
    m) MEMORIZER_ID=$OPTARG ;;
    o) OUTPUT_STYLE=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

# Check required parameters
if [ -z "$GPU_ID" ] || [ -z "$PORT" ]; then
  echo "Usage: $0 -g <GPU_ID> -p <PORT> [-c <commander>] [-m <memorizer>] [-o <output>]"
  exit 1
fi

CONTAINER_NAME="magma_agent_gpu_${GPU_ID}"
IMAGE_NAME="magma_agent_image"

# Build the image
docker build -t "${IMAGE_NAME}" .

# Run container
docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU_ID}" \
    -e COMMANDER_ID="${COMMANDER_ID}" \
    -e MEMORIZER_ID="${MEMORIZER_ID}" \
    -e COMMANDER_OUTPUT_STYLE="${OUTPUT_STYLE}" \
    -e QWEN_QUANTIZATION="${QWEN_QUANTIZATION:-4bit}" \
    -e QWEN_MAX_BATCH_SIZE="${QWEN_MAX_BATCH_SIZE:-1}" \
    -e QWEN_MAX_NEW_TOKENS="${QWEN_MAX_NEW_TOKENS:-1500}" \
    -e QWEN_ATTN_IMPLEMENTATION="${QWEN_ATTN_IMPLEMENTATION:-sdpa}" \
    -e QWEN_USE_CACHE="${QWEN_USE_CACHE:-true}" \
    -e QWEN_DEVICE_MAP="${QWEN_DEVICE_MAP:-cuda}" \
    -e QWEN_GPU_MEMORY_LIMIT="${QWEN_GPU_MEMORY_LIMIT:-}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e CONTAINER_NAME="instance_${GPU_ID}.log" \
    -p ${PORT}:8888 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${IMAGE_NAME}"
