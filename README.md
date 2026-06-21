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

## Model configuration

`magma-agent` loads models from a single multi-model configuration. Provide it
with `MAGMA_MODELS_CONFIG`, `MAGMA_MODELS_JSON`, `--models-config`, or
`--models-json`.

```json
{
  "models": [
    {
      "name": "main_commander",
      "type": "Commander",
      "model_id": "path-or-hf-id",
      "endpoint": "/chat",
      "options": {
        "output_style": "qwen_format"
      }
    },
    {
      "name": "task_state_manager",
      "type": "TSM",
      "model_id": "path-or-hf-id",
      "endpoint": "/update_task_state"
    }
  ]
}
```

`/get_infos` returns the loaded models as a `models` list with `name`, `type`,
`endpoint`, and `model_id`.

## Large Qwen models

Qwen commander options are declared inside the model `options` object:

- `quantization_mode`: `4bit`, `8bit`, or `none`
- `max_new_tokens`: default `1500`
- `attn_implementation`: `sdpa`, `flash_attention_2`, or `eager`
- `use_cache`: set to `false` as a last-resort memory tradeoff
- `device_map`: `auto`, `cuda`, or `cpu`
- `gpu_memory_limit`: for example `42GiB`
- `allow_cpu_offload`: `true` or `false`

For a 27B model on a 48 GB GPU, the Qwen path uses `device_map=auto` by
default and leaves GPU headroom during loading to avoid out-of-memory spikes.
If quality loss from 4-bit NF4 is too high, try `QWEN_QUANTIZATION=8bit`
with the same loading settings.


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
