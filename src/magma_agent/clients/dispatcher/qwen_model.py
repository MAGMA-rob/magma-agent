from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch  # type: ignore

from magma_agent.clients.commander.qwen_model import (
    _best_compute_dtype,
    _build_load_kwargs,
    _build_quantization_config,
    _log_cuda_memory,
    _log_loaded_model_state,
)

from .base import BaseDispatcher
from .messages import (
    BatchedMessageDispatcher,
    REPRESENTATION_FIELDS,
    format_dispatcher_history,
    get_permanent_rules,
    get_representation_field,
)
from .parsing import parse_dispatcher_output


BASE_SYSTEM_PROMPT = """You are MAGMA's robot dispatcher.

Your job is to choose the next useful actions to progress the current task.

You are given:

* Rules that must always be respected.
* Permanent rules containing immutable guidance that must never be changed.
* A todo list describing the remaining work.
* Task attributes describing the current environment.
* A compact execution history containing previous tool calls and their results.
* The list of available tools.

## Guidelines

* Use only the provided tools.
* Every tool call must select exactly one robot from `known_robots`.
* Generate valid JSON arguments that match the tool schema.
* Ground every argument in the provided context.
* Continue the current todo until it is completed.
* Recover from previous execution failures whenever possible.
* Ask the user for clarification instead of guessing when required information is missing.

## Output

Output **only** a JSON object.

The object may contain:
- Either a list of tools **OR** a message
- Zero or more completed todos

### `tools`

A list of tool calls to execute.

Each tool call has the following format:

```json
{
  "robot": "robot_name",
  "name": "tool_name",
  "arguments": {
    "parameter": "value"
  }
}
```

### `message`

Use this field when todo requires 'Inform' or 'Answer' or 'Find' to say/ask something to the user.


```json
{
  "recipient": "user",
  "content": "..."
}
```

or use this field when you have finished all the todo or encountering blocking issues to inform the system.

```json
{
  "recipient": "system",
  "content": "..."
}
```

### `completed_todos`

A list of completed todo identifiers.

```json
[
  "0",
  "1"
]
```

## Output rules

* Return only a JSON object.
* Do not output markdown or code fences.
* Do not output any text outside the JSON object.
* `tools` and `message` are mutually exclusive. Never output both in the same response.
* `tools` may contain one, or multiple tool calls.
* `message` may contain at most one message.
* Do not generate fields other than `tools`, `message`, and `completed_todos`.
* Do not expose reasoning, analysis, or chain of thought.
* If dispatching a tool can make progress, dispatch a tool instead of sending a message.
* `completed_todos` must only be used for a todo that you have completed, not that you are launching. Meaning you must wait for the tool return before calling it.

## Valid exemples:
```json
{
    "message":{...},
    "completed_todos":[...]
}
```
```json
{
    "tools": [{...}],
    "completed_todos":[...]
}
```

Never output markdown or code fences.

/no_think
"""


class QwenDispatcher(BaseDispatcher):

    def __init__(
        self,
        model_name: str,
        cpu_load: bool = False,
        name: str = "dispatcher",
        quantization_mode: str = "4bit",
        max_new_tokens: int = 600,
        attn_implementation: Optional[str] = "sdpa",
        use_cache: bool = True,
        device_map: str = "auto",
        gpu_memory_limit: Optional[str] = None,
        allow_cpu_offload: bool = False,
        offload_folder: str = "/tmp/magma_agent_qwen_dispatcher_offload",
        enable_thinking: bool = False,
    ) -> None:
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.use_cache = use_cache
        self.enable_thinking = bool(enable_thinking)

        compute_dtype = _best_compute_dtype()
        quantization = _build_quantization_config(
            quantization_mode,
            compute_dtype,
            allow_cpu_offload=allow_cpu_offload,
        )
        load_kwargs = _build_load_kwargs(
            attn_implementation=attn_implementation,
            device_map=device_map,
            gpu_memory_limit=gpu_memory_limit,
            offload_folder=offload_folder,
        )

        print(
            "[QWEN DISPATCHER] Loading with "
            f"quantization={quantization_mode}, compute_dtype={compute_dtype}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"use_cache={self.use_cache}, allow_cpu_offload={allow_cpu_offload}, "
            f"enable_thinking={self.enable_thinking}, "
            f"load_kwargs={load_kwargs}"
        )
        _log_cuda_memory("before Qwen dispatcher load")
        super().__init__(
            model_name,
            cpu_load=False,
            name=name,
            dtype=compute_dtype,
            quantization=quantization,
            load_kwargs=load_kwargs,
            runtime_device_move=False,
        )
        default_template_path = (
            Path(__file__).resolve().parents[2]
            / "default_chat_template"
            / "dispatcher.jinja"
        )
        self.tokenizer.chat_template = default_template_path.read_text(
            encoding="utf-8"
        )
        _log_loaded_model_state(self.model)
        _log_cuda_memory("after Qwen dispatcher load")
        if cpu_load:
            print(
                "[QWEN DISPATCHER] optimize_memory CPU offload is disabled for "
                "quantized/device-mapped Qwen. Use max_new_tokens to reduce runtime memory."
            )

    def process_batched_entry(
        self,
        message: BatchedMessageDispatcher,
        inference_mode: bool,
    ) -> List[Union[Dict[str, Any], str]]:
        formatted_inputs = []
        batch_size = len(message.memory)

        if not batch_size:
            raise ValueError(
                "BatchedMessageDispatcher must contain at least one entry."
            )

        for field_name in ("attributes", "history", "function"):
            field_value = getattr(message, field_name)
            if len(field_value) != batch_size:
                raise ValueError(
                    f"{field_name} must have the same batch length "
                    f"({len(field_value)} != {batch_size})."
                )

        for i in range(batch_size):
            memory = message.memory[i]
            representation = {
                field_name: get_representation_field(memory, field_name)
                for field_name in REPRESENTATION_FIELDS
            }
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [{}],
                    tools=message.function[i],
                    task_attributes=message.attributes[i],
                    permanent_rules=get_permanent_rules(memory),
                    rules=representation["rules"],
                    todo=representation["todo"],
                    history=format_dispatcher_history(message.history[i]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.input_device)
        prompt_length = inputs["input_ids"].shape[1]
        generation_kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": self.max_new_tokens,
            "use_cache": self.use_cache,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.inference_mode():
            if inference_mode:
                output = self.model.generate(
                    **generation_kwargs,
                    do_sample=False,
                )
            else:
                output = self.model.generate(
                    **generation_kwargs,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.85,
                    top_k=20,
                )

        responses = []
        for i in range(len(formatted_inputs)):
            if i==0:
                print("--------") 
                print(self.tokenizer.decode(
                    output[i],
                    skip_special_tokens=True,
                ).strip())
            generated_tokens = output[i][prompt_length:]
            response_text = self.tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()
            parsed_response = parse_dispatcher_output(response_text)
            self.log_prompt_exchange(
                formatted_inputs[i],
                response_text,
                isinstance(parsed_response, dict),
            )
            responses.append(parsed_response)

        return responses
