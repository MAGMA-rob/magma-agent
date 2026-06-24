from typing import Dict, List, Optional
import torch
import json, os, re

from .messages import BatchedMessageCommander, get_memory_list
from .base import BaseCommander
from .history import get_history_content, get_instruction_roles, map_chat_role

TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
UNFINISHED_TOOL_RE = re.compile(r"<tool_call>\s*(.*)$", re.DOTALL)


class MagmaCommander(BaseCommander):

    def __init__(
        self,
        model_id,
        output_style,
        overriding_chat_template_path: Optional[str],
        cpu_load: bool,
        name: str = "commander",
        endpoint: str = "/chat",
    ) -> None:
        super().__init__(model_id, cpu_load, name=name, endpoint=endpoint)
        if overriding_chat_template_path is not None:
            with open(overriding_chat_template_path, "r", encoding="utf-8") as f:
                chat_template_content = f.read()
            self.tokenizer.chat_template = chat_template_content

        if output_style != "json" and output_style != "qwen_format":
            raise ValueError(f"Unknow {output_style} commander output format. Avalaibles are json or qwen_format")
        self.output_style = output_style

    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
 
        mx_lenght = 0
        formatted_inputs = []
        instruction_roles = get_instruction_roles(message)
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

        for i in range(batch_size):
            memory = "Memory:\n" 
            for mem in get_memory_list(message.memory[i]):
                memory += f"- {mem}\n"

            messages= []

            for previous_mess in message.history[i]:
                messages.append({
                    "role": map_chat_role(
                        previous_mess.get("author"),
                        system_role="status",
                        model_role="model",
                    ),
                    "content": get_history_content(previous_mess),
                })

            messages.append({
                "role": map_chat_role(
                    instruction_roles[i],
                    system_role="status",
                    model_role="model",
                ),
                "content": message.instruction[i],
            })
            formatted_inputs.append(self.tokenizer.apply_chat_template(
                    messages,
                    tools=message.function[i],
                    long_memory=memory,
                    task_attributes=message.attributes[i],
                    tokenize=False,
                    add_generation_prompt=True
                )
            )
            lenght = len(self.tokenizer(formatted_inputs[i], return_tensors="pt")["input_ids"][0])
            if lenght == 0:
                raise ValueError(
                    "The commander chat template produced an empty prompt. "
                    "Check that the loaded tokenizer/chat_template matches the "
                    "model and the MagmaCommander formatting arguments."
                )

            if mx_lenght < lenght:
                mx_lenght = lenght

        inputs = self.tokenizer(formatted_inputs, return_tensors="pt", padding=True).to(self.input_device)
        if os.getenv("MAGMA_DEBUG_TOKENIZER") == "1":
            print("[COMMANDER][BATCH TOKENIZER DEBUG]")
            print(f"batch_input_ids_shape={tuple(inputs['input_ids'].shape)}")
            if "attention_mask" in inputs:
                print(f"attention_mask_sums={inputs['attention_mask'].sum(dim=1).tolist()}")
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

            if self.output_style == "json":
                try:
                    responses.append(json.loads(response_text))
                except:
                    print(f"[COMMANDER] Bad model output")
                    responses.append({"think":"", "say":"", "action":response_text})
            else:
                st = parse_blocks(response_text)
                responses.append(st)

        return responses


def parse_blocks(text):
    # 1. extract think
    think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    think = think_match.group(1).strip() if think_match else ""

    # 2. remove think block to isolate the rest
    after_think = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 3. extract tool_call JSON
    tool_match = TOOL_RE.search(after_think) or UNFINISHED_TOOL_RE.search(after_think)
    if tool_match:
        tool_call = tool_match.group(1).strip()
        try:
            action = json.loads(tool_call)
        except json.JSONDecodeError:
            print("[COMMANDER] BAD MODEL OUTPUT")
            action = tool_call

        # 4. extract the "say" text (everything between </think> and <tool_call>)
        say = _strip_terminal_tokens(TOOL_RE.sub("", after_think))
        say = _strip_terminal_tokens(UNFINISHED_TOOL_RE.sub("", say))
    else:
        # Backward-compatible recovery for checkpoints that output: say + raw JSON.
        action, say = _split_raw_trailing_json(after_think)

    return {
        "think": think,
        "say": say,
        "action": action,
    }


def _split_raw_trailing_json(text):
    decoder = json.JSONDecoder()
    parsed_candidates = []

    for match in re.finditer(r"\{", text):
        start = match.start()
        try:
            parsed, offset = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            parsed_candidates.append((start, start + offset, parsed))

    if not parsed_candidates:
        marker = re.search(r"\btool_call\s*:\s*", text, flags=re.IGNORECASE)
        if marker:
            say = text[:marker.start()].strip()
            return text[marker.end():].strip(), say
        return {}, text.strip()

    clean_terminal_candidates = [
        candidate
        for candidate in parsed_candidates
        if _is_terminal_suffix(text[candidate[1]:])
    ]
    if clean_terminal_candidates:
        start, end, action = min(clean_terminal_candidates, key=lambda candidate: candidate[0])
    else:
        start, end, action = max(
            parsed_candidates,
            key=lambda candidate: (candidate[1] - candidate[0], -candidate[0]),
        )
    suffix = text[end:]
    if _is_terminal_suffix(suffix):
        suffix = ""
    say = _strip_terminal_tokens(text[:start] + suffix)
    say = re.sub(r"\btool_call\s*:\s*$", "", say, flags=re.IGNORECASE).strip()
    say = re.sub(r"^\s*say\s*:\s*", "", say, flags=re.IGNORECASE).strip()
    return action, say


def _is_terminal_suffix(text):
    suffix = text.strip()
    if not suffix:
        return True
    return suffix in {"</s>", "<|im_end|>", "<|endoftext|>"}


def _strip_terminal_tokens(text):
    text = text.strip()
    for token in ("</s>", "<|im_end|>", "<|endoftext|>"):
        if text.endswith(token):
            text = text[: -len(token)].strip()
    return text
 
    
