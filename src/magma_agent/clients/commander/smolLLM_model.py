from typing import Dict, List, Optional
import torch
import json, os, re

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander

SINGLE_AGENT = """
You are a controlling a robot through optional tool calls.

You must follow user instruction and call only ONE tool in <tool_call>...</tool_call>

What you are writing outside the <tool_call> block will be say to the user.
Keep it concice and use it to answer to the user, and show your understanding of differents information.

Rules:
- Do NOT put JSON outside <tool_call>.
- Only if an action is required.
- The tool call must be valid JSON with EXACT schema : {"name":<func_name>, "arguments":{"param":<value>...}}
- Only ONE tool call is possible.
/no_think
"""

BASE_SYSTEM_PROMPT = """You are a COMMANDER agent controlling a robot through optional tool calls.
You operate in a long-horizon task, but you do NOT manage persistent memory yourself.

A separate agent (the Memorizer) will handle memory updates.
Your role is to make the task state EXPLICIT so another agent can maintain memory correctly.

You will receive three structured inputs:
1. "memory": a persistent task memory summarizing past decisions, constraints, and commitments. Treat it as authoritative.
2. "task_attributes": structured metadata describing the current task state.
3. "query": the current user request or system update.

You must interpret the query using both memory and task_attributes.

Output MUST follow this structure, in this exact order:

<intent> ... </intent>
<say> ... </say>
<tool_call> ... </tool_call>

Here are some rules to respect when filling each block:
intent :
   - Short, explicit description of:
     - user intent
     - constraints. You must fully note them like 'default assignment is X to Y".
     - current subgoal
     - what must be remembered, updated, or forgotten
   - Written for another agent.
   - Must be understandable without context.

say :
   - What is said to the user.

tool_call :
   - Only if an action is required.
   - Must be valid JSON with EXACT schema : {"name":<func_name>, "arguments":{"param":<value>...}}
   - Only tool call is possible. If you need more, you can expliclty say that you need to call them later.

Rules:
- Do NOT put JSON outside <tool_call>.
- Do NOT explain the structure.
- Do NOT merge fields.

Rules (mandatory):
1. No text outside the provided structure.
2. Never omit any of the four top-level fields.
3. Never call more than one tool.
4. Only call a tool when it is clearly required by the task logic.
5. The "intent" field must always be present and explicit.
6. Do NOT manage or modify memory directly.
7. If memory contradicts the query, memory overrides the query.
8. If the user is simply giving constraints, you must acknowledge them explictly in the say.
/no_think
"""


class SmolLMCommander(BaseCommander):

    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool, dual_mode : bool = True) -> List[Dict]:
 
        mx_lenght = 0
        formatted_inputs = []

        for i in range(len(message.memory)):
            memory = "Memory:\n" 
            for mem in message.memory[i]:
                memory += f"- {mem}\n"

            
            messages= [{"role":"system","content":BASE_SYSTEM_PROMPT if dual_mode else SINGLE_AGENT}]

            for previous_mess in message.history[i]:
                messages.append({"role" : previous_mess.get("author", "user"), "content": previous_mess.get("sentence", "")})

            messages.append({"role": "user", "content": message.instruction[i]})
            
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
                    max_new_tokens=2500,
                    do_sample=False,  
                )
            else:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=2500,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20
                )
        responses = []
        for i in range(len(formatted_inputs)):
            generated_tokens = output[i][input_lengths[i]:]
            response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            if dual_mode:
                st = parse_dual(response_text)
            else:
                st = parse_blocks(response_text)
            print(st)
            responses.append(st)

        return responses


def parse_blocks(text):
    # 1. extract think
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    # 2. remove think block to isolate the rest
    after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 3. extract tool_call JSON
    tool_match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", after_think, re.DOTALL)
    tool_call = tool_match.group(1).strip() if tool_match else "{}"
    try:
        action = json.loads(tool_call)
    except:
        print("[COMMANDER] BAD MODEL OUTPUT")
        action = tool_call

    # 4. extract the "say" text (everything between </think> and <tool_call>)
    say = re.sub(r"<tool_call>.*?</tool_call>", "", after_think, flags=re.DOTALL).strip()

    return {
        "think": think,
        "say": say,
        "action": action,
    }
 

THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
INTENT_RE = re.compile(r"<intent>\s*(.*?)\s*</intent>", re.DOTALL)
SAY_RE = re.compile(r"<say>\s*(.*?)\s*</say>", re.DOTALL)
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)  

def parse_dual(text):
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
 