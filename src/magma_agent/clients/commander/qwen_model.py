from typing import Dict, List, Optional
import torch
import json, os, re

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

    def __init__(self, model_name : str, cpu_load: bool = False) -> None:
        quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        super().__init__(model_name, cpu_load=False, quantization= quantization)

    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
        system_message = {'role': "system", "content":BASE_SYSTEM_PROMPT}
        formatted_inputs = []
        mx_lenght = 0
        instruction_roles = get_instruction_roles(message)

        for i in range(len(message.instruction)):
            
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

            lenght = len(self.tokenizer(formatted_inputs[i], return_tensors="pt")["input_ids"][0])

            if mx_lenght < lenght:
                mx_lenght = lenght

        inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding="max_length", max_length=mx_lenght).to(self.model.device)
        input_lengths = [len(x) for x in inputs["input_ids"]]
        with torch.no_grad():
            if inference_mode:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    do_sample=False,  
                )
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=1500,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20
                )
        responses = []
        for i in range(len(formatted_inputs)):
            generated_tokens = output[i][input_lengths[i]:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            st = parse_blocks(response_text)
            responses.append(st)

        return responses

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
INTENT_RE = re.compile(r"<intent>\s*(.*?)\s*</intent>", re.DOTALL)
SAY_RE = re.compile(r"<say>\s*(.*?)\s*</say>", re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)  

def parse_blocks(text):
    think_match = THINK_RE.search(text)
    intent_match = INTENT_RE.search(text)
    say_match = SAY_RE.search(text)
    tool_match = TOOL_RE.search(text)

    think = think_match.group(1).strip() if think_match else ""
    intent = intent_match.group(1).strip() if intent_match else ""
    say = say_match.group(1).strip() if say_match else ""

    # --- parse tool call ---
    action = {}
    if tool_match:
        raw = tool_match.group(1).strip()
        try:
            action = json.loads(raw)
        except json.JSONDecodeError:
            # hard failure: tool call must be exact
            print("[PARSE ERROR] Invalid tool_call JSON")
            action = {}

    # --- sanity checks (optional but recommended) ---
    if not intent:
        print("[WARNING] Missing <intent> block")

    if not say:
        print("[WARNING] Missing <say> block")

    return {
        "reasoning": think,
        "think": intent,
        "say": say,
        "action": action,
    }
 
