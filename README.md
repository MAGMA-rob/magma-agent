# MAGMA_Agent
![Docker](https://img.shields.io/badge/Docker-Supported-blue)
![Ubuntu version](https://img.shields.io/badge/Ubuntu-24.04-blue)

This repository contains a python package to run a fastapi/uvicorn server in python to deploy an agent into the MAGMA-GEN pipeline, MAGMA-BENCH evaluation and MAGMA-ROS2 applications.

<p align="center">
  <b>⚠️ EXPERIMENTAL PROJECT — DOCUMENTATION IN PROGRESS ⚠️</b>
</p>

### 🚧 Documentation Status
<div style="padding:10px; border-left:4px solid red; background-color:#fff5f5;"> <strong>📌 Documentation to do</strong><br> This README is intentionally simplified. Full documentation and usage examples will be added soon. </div>

## Getting started

You can use this package to define your own agents script or simply to use an already supported model.

## Use this package with your own model

Check documentation

## Large Qwen models

Original Qwen commanders are loaded through a separate memory path so Magma
commanders keep the default behavior. The useful knobs are:

- `QWEN_QUANTIZATION=4bit|8bit|none` or `--qwen-quantization`
- `QWEN_MAX_BATCH_SIZE=1` or `--qwen-max-batch-size`
- `QWEN_MAX_NEW_TOKENS=1500` or `--qwen-max-new-tokens`
- `QWEN_ATTN_IMPLEMENTATION=sdpa|flash_attention_2|eager` or `--qwen-attn-implementation`
- `QWEN_USE_CACHE=false` or `--no-qwen-use-cache` as a last-resort memory tradeoff
- `QWEN_DEVICE_MAP=auto|cuda|cpu` or `--qwen-device-map`
- `QWEN_GPU_MEMORY_LIMIT=42GiB` or `--qwen-gpu-memory-limit`
- `QWEN_ALLOW_CPU_OFFLOAD=true` or `--qwen-allow-cpu-offload`

For a 27B model on a 48 GB GPU, start with `QWEN_MAX_BATCH_SIZE=1`.
The Qwen path uses `device_map=auto` by default and leaves GPU headroom during
loading to avoid out-of-memory spikes.
If quality loss from 4-bit NF4 is too high, try `QWEN_QUANTIZATION=8bit`
with the same micro-batch size.


## Support
You can contact me at l.bernat@sileane.com

## To do:
- Gpt-oss official support

## Known bug
- Memory leak. The memory growth slowly as request arrives.

## Authors and acknowledgment
Loan BERNAT (l.bernat@sileane.com)

## License
BSD 2 clauses
