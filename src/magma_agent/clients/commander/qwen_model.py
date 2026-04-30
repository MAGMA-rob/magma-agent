from typing import Any, Dict, List, Optional
import torch
import json, re
from pathlib import Path

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander
from .history import get_history_content, get_instruction_roles, map_chat_role

from transformers import BitsAndBytesConfig

BASE_SYSTEM_PROMPT = """You are a COMMANDER agent controlling a robot through structured tool calls.

Your role is to decide the NEXT best action given:
- the current user query
- the task attributes
- the available tools
- the full interaction history

You must output TWO things:
1. A natural language response to the user
2. (Optional) ONE tool call

----------------------------------------
OUTPUT FORMAT (STRICT)
----------------------------------------

say:
- What you say to the user

<tool_call>
{"robot_name":{"name":<func_name>, "arguments":{"param":<value>}}}
</tool_call>

If no action is needed, DO NOT output <tool_call>.

----------------------------------------
CORE DECISION RULES
----------------------------------------

1. HISTORY-AWARE DECISION
- Use the full interaction history to infer:
  - what has already been done
  - what failed or succeeded
  - what remains to be achieved
- Do NOT repeat actions that already succeeded
- If an action may have failed, consider re-checking before retrying

2. LONG-HORIZON CONSISTENCY
- Your goal is NOT to complete the task in one step
- You must maintain a coherent multi-step strategy
- Prefer safe intermediate actions over risky assumptions

3. PARTIAL OBSERVABILITY
- The environment may be incomplete or outdated
- Do NOT assume objects are present unless confirmed
- Use perception tools (e.g. detect) when needed

4. EXECUTION UNCERTAINTY
- A correct action can still fail
- If an action might have failed:
  - verify before continuing
  - retry if appropriate

5. TOOL USAGE POLICY
- Only call a tool if it is necessary for progress
- Only ONE tool call per step
- Arguments must be grounded in known objects or attributes
- Do NOT hallucinate object names

6. NO OVER-COMMITMENT
- If information is missing:
  - ask for clarification OR
  - call a perception tool
- Do NOT guess hidden state

----------------------------------------
TOOL CALL FORMAT (STRICT)
----------------------------------------

- Must be valid JSON
- EXACT schema:
  {"robot_name":{"name":<func_name>, "arguments":{"param":<value>}}}

- NO extra fields
- NO comments
- NO text outside <tool_call>

----------------------------------------
BEHAVIORAL GUIDELINES
----------------------------------------

- Be concise and goal-directed
- Do not explain your reasoning
- Do not describe the schema
- Do not output multiple tool calls
- Do not simulate future steps

----------------------------------------
FAILURE HANDLING
----------------------------------------

If the situation is uncertain or inconsistent:
- Prefer VERIFY → then ACT
- Prefer RECOVER → instead of restarting

----------------------------------------
IMPORTANT
----------------------------------------

You are trained to operate under:
- partial observability
- stochastic execution
- evolving goals

Your objective is to produce actions that remain consistent and recoverable over time, not just locally optimal.

----------------------------------------
INPUTS
----------------------------------------
"""


class QwenCommander(BaseCommander):

    def __init__(
        self,
        model_name : str,
        cpu_load: bool = False,
        quantization_mode: str = "4bit",
        max_new_tokens: int = 1500,
        attn_implementation: Optional[str] = "sdpa",
        use_cache: bool = True,
        device_map: str = "auto",
        gpu_memory_limit: Optional[str] = None,
        allow_cpu_offload: bool = False,
        offload_folder: str = "/tmp/magma_agent_qwen_offload",
    ) -> None:
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.use_cache = use_cache

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
            "[QWEN] Loading with "
            f"quantization={quantization_mode}, compute_dtype={compute_dtype}, "
            f"max_new_tokens={self.max_new_tokens}, "
            f"use_cache={self.use_cache}, allow_cpu_offload={allow_cpu_offload}, "
            f"load_kwargs={load_kwargs}"
        )
        _log_cuda_memory("before Qwen load")
        super().__init__(
            model_name,
            cpu_load=False,
            dtype=compute_dtype,
            quantization=quantization,
            load_kwargs=load_kwargs,
            runtime_device_move=False,
        )
        _log_loaded_model_state(self.model)
        _log_cuda_memory("after Qwen load")
        if cpu_load:
            print(
                "[QWEN] optimize_memory CPU offload is disabled for quantized/device-mapped Qwen. "
                "Use QWEN_MAX_NEW_TOKENS to reduce runtime memory."
            )

    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
        system_message = {'role': "system", "content":BASE_SYSTEM_PROMPT}
        formatted_inputs = []
        instruction_roles = get_instruction_roles(message)
        batch_size = len(message.instruction)

        _validate_batch(message)

        for i in range(batch_size):
            
            mem_str = "Memory:\n"
            if len(message.memory[i]) > 0:
                for mem in message.memory[i]:
                    mem_str += f"- {mem}\n"
            else:
                mem_str += "empty\n"

            prompt_user = f"Task Attributes : {message.attributes[i]}.\n{mem_str}\nQuery : {message.instruction[i]}"

            messages = [
                system_message.copy()
            ]
            for previous_mess in message.history[i]:
                messages.append({
                    "role": map_chat_role(previous_mess.get("author")),
                    "content": get_history_content(previous_mess),
                })

            messages.append({
                "role": map_chat_role(instruction_roles[i]),
                "content": prompt_user,
            })

            formatted_inputs.append(self.tokenizer.apply_chat_template(
                    messages,
                    tools=message.function[i],
                    tokenize=False,
                    add_generation_prompt=True
                )
            )

        inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(self.input_device)
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
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20
                )
        responses = []
        for i in range(len(formatted_inputs)):
            generated_tokens = output[i][prompt_length:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            st = parse_blocks(response_text)
            responses.append(st)

        return responses


def _best_compute_dtype() -> torch.dtype:
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _build_quantization_config(
    quantization_mode: str,
    compute_dtype: torch.dtype,
    allow_cpu_offload: bool,
):
    mode = (quantization_mode or "4bit").lower()
    if mode in ("4bit", "4-bit", "nf4"):
        return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                llm_int8_enable_fp32_cpu_offload=allow_cpu_offload,
            )
    if mode in ("8bit", "8-bit", "int8"):
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=allow_cpu_offload,
        )
    if mode in ("none", "no", "false", "0"):
        return None
    raise ValueError("qwen_quantization must be one of: 4bit, 8bit, none")


def _build_load_kwargs(
    attn_implementation: Optional[str],
    device_map: str,
    gpu_memory_limit: Optional[str],
    offload_folder: str,
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    normalized_device_map = (device_map or "auto").lower()
    if normalized_device_map == "auto":
        kwargs["device_map"] = "auto"
    elif normalized_device_map == "cpu":
        kwargs["device_map"] = {"": "cpu"}
    elif normalized_device_map == "cuda":
        if torch.cuda.is_available():
            kwargs["device_map"] = {"": torch.cuda.current_device()}
        else:
            kwargs["device_map"] = {"": "cpu"}
    else:
        raise ValueError("qwen_device_map must be one of: cuda, auto, cpu")

    if torch.cuda.is_available() and normalized_device_map == "auto":
        kwargs["max_memory"] = {torch.cuda.current_device(): gpu_memory_limit or _default_gpu_memory_limit()}

    if normalized_device_map == "auto" and offload_folder:
        Path(offload_folder).mkdir(parents=True, exist_ok=True)
        kwargs["offload_folder"] = offload_folder
        kwargs["offload_buffers"] = True

    return kwargs


def _default_gpu_memory_limit() -> str:
    free_bytes, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    # Leave headroom for bnb conversion buffers and allocator fragmentation during from_pretrained.
    limit_gib = max(1, int((free_bytes * 0.88) // 1024**3))
    return f"{limit_gib}GiB"


def _log_cuda_memory(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[QWEN][MEM] {label}: cuda unavailable")
        return

    index = torch.cuda.current_device()
    free_bytes, total_bytes = torch.cuda.mem_get_info(index)
    allocated = torch.cuda.memory_allocated(index)
    reserved = torch.cuda.memory_reserved(index)
    print(
        f"[QWEN][MEM] {label}: "
        f"free={free_bytes / 1024**3:.2f}GiB total={total_bytes / 1024**3:.2f}GiB "
        f"allocated={allocated / 1024**3:.2f}GiB reserved={reserved / 1024**3:.2f}GiB"
    )


def _log_loaded_model_state(model: Any) -> None:
    device_map = getattr(model, "hf_device_map", None)
    quantization_method = getattr(model, "quantization_method", None)
    is_loaded_in_4bit = getattr(model, "is_loaded_in_4bit", False)
    is_loaded_in_8bit = getattr(model, "is_loaded_in_8bit", False)
    print(
        "[QWEN] Loaded model state: "
        f"is_loaded_in_4bit={is_loaded_in_4bit}, "
        f"is_loaded_in_8bit={is_loaded_in_8bit}, "
        f"quantization_method={quantization_method}, "
        f"device_map={device_map}"
    )
    if hasattr(model, "get_memory_footprint"):
        try:
            footprint = model.get_memory_footprint()
            print(f"[QWEN] Model memory footprint={footprint / 1024**3:.2f}GiB")
        except Exception as err:
            print(f"[QWEN] Could not compute model memory footprint: {err}")


def _validate_batch(message: BatchedMessageCommander) -> None:
    batch_size = len(message.instruction)
    if not batch_size:
        raise ValueError("BatchedMessageCommander must contain at least one instruction.")

    for field_name in ("memory", "attributes", "history", "function"):
        field_value = getattr(message, field_name)
        if len(field_value) != batch_size:
            raise ValueError(
                f"{field_name} must have the same length as instruction "
                f"({len(field_value)} != {batch_size})."
            )


TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)  

def parse_blocks(text):
    tool_match = TOOL_RE.search(text)

    action = {}
    say = ""

    if tool_match:
        # --- extract tool ---
        raw = tool_match.group(1).strip()
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            print("[PARSE ERROR] Invalid tool_call JSON")
            action = {}

        # --- extract say (everything before tool_call) ---
        say = text[:tool_match.start()].strip()

    else:
        # no tool_call → everything is say
        say = text.strip()

    return {
        "think": "-",  # volontairement ignoré
        "say": say,
        "action": action,
    }
 
