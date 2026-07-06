#!/bin/bash
set -e

GPU_ID=""
PORT=""
AGENT_CONFIG=""
AGENT_JSON=""

while getopts "g:p:f:j:" opt; do
  case $opt in
    g) GPU_ID=$OPTARG ;;
    p) PORT=$OPTARG ;;
    f) AGENT_CONFIG=$OPTARG ;;
    j) AGENT_JSON=$OPTARG ;;
    *) echo "Invalid option"; exit 1 ;;
  esac
done

if [ -z "$GPU_ID" ] || [ -z "$PORT" ]; then
  echo "Usage: $0 -g <GPU_ID> -p <PORT> (-f <agent_config.json> | -j <agent_json>)"
  exit 1
fi

if [ -z "$AGENT_CONFIG" ] && [ -z "$AGENT_JSON" ]; then
  echo "An agent config is required. Use -f <agent_config.json> or -j <agent_json>."
  exit 1
fi

if [ -n "$AGENT_CONFIG" ] && [ -n "$AGENT_JSON" ]; then
  echo "Use either -f or -j, not both."
  exit 1
fi

CONTAINER_NAME="magma_agent_gpu_${GPU_ID}"
IMAGE_NAME="magma_agent_image"
CONFIG_MOUNT_ARGS=()
CONFIG_ENV_ARGS=()

if [ -n "$AGENT_CONFIG" ]; then
  CONFIG_PATH="$(cd "$(dirname "$AGENT_CONFIG")" && pwd)/$(basename "$AGENT_CONFIG")"
  CONFIG_MOUNT_ARGS=(-v "${CONFIG_PATH}:/app/agent_config.json:ro")
  CONFIG_ENV_ARGS=(-e CONFIG_PATH="/app/agent_config.json")
else
  CONFIG_ENV_ARGS=(-e CONFIG_JSON="${AGENT_JSON}")
fi

docker build -t "${IMAGE_NAME}" .

docker run --rm \
    --name "${CONTAINER_NAME}" \
    --gpus "device=${GPU_ID}" \
    "${CONFIG_ENV_ARGS[@]}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e LOG_FILE="logs/instance_${GPU_ID}.log" \
    -e PROMPT_LOG_DIR="${PROMPT_LOG_DIR:-logs/prompts}" \
    -p ${PORT}:8888 \
    -v "$(pwd)/models:/app/models" \
    -v "$(pwd)/logs:/app/logs" \
    "${CONFIG_MOUNT_ARGS[@]}" \
    "${IMAGE_NAME}"
