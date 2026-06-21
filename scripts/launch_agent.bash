#!/bin/bash
set -e

GPU_ID=""
PORT=""
MODELS_CONFIG=""
MODELS_JSON=""

while getopts "g:p:f:j:" opt; do
  case $opt in
    g) GPU_ID=$OPTARG ;;
    p) PORT=$OPTARG ;;
    f) MODELS_CONFIG=$OPTARG ;;
    j) MODELS_JSON=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ] || [ -z "$PORT" ]; then
  echo "Usage: $0 -g <GPU_ID> -p <PORT> (-f <models_config.json> | -j <models_json>)"
  exit 1
fi

if [ -z "$MODELS_CONFIG" ] && [ -z "$MODELS_JSON" ]; then
  echo "A model config is required. Use -f <models_config.json> or -j <models_json>."
  exit 1
fi

if [ -n "$MODELS_CONFIG" ] && [ -n "$MODELS_JSON" ]; then
  echo "Use either -f or -j, not both."
  exit 1
fi

CONTAINER_NAME="magma_agent_gpu_${GPU_ID}"
IMAGE_NAME="magma_agent_image"
CONFIG_MOUNT_ARGS=()
CONFIG_ENV_ARGS=()

if [ -n "$MODELS_CONFIG" ]; then
  CONFIG_PATH="$(cd "$(dirname "$MODELS_CONFIG")" && pwd)/$(basename "$MODELS_CONFIG")"
  CONFIG_MOUNT_ARGS=(-v "${CONFIG_PATH}:/app/models_config.json:ro")
  CONFIG_ENV_ARGS=(-e MAGMA_MODELS_CONFIG="/app/models_config.json")
else
  CONFIG_ENV_ARGS=(-e MAGMA_MODELS_JSON="${MODELS_JSON}")
fi

docker build -t "${IMAGE_NAME}" .

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU_ID}" \
    "${CONFIG_ENV_ARGS[@]}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e LOG_FILE="logs/instance_${GPU_ID}.log" \
    -p ${PORT}:8888 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${CONFIG_MOUNT_ARGS[@]}" \
    "${IMAGE_NAME}"
