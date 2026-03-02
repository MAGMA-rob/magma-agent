from typing import Dict, List, Optional
import torch
import json, os, re

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander

class MagmaCommander(BaseCommander):

    def __init__(self, model_id, output_style, overriding_chat_template_path : Optional[str], cpu_load : bool) -> None:
        super().__init__(model_id, cpu_load)
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

        for i in range(len(message.memory)):
            memory = "Memory:\n" 
            for mem in message.memory[i]:
                memory += f"- {mem}\n"

            messages= []

            for previous_mess in message.history[i]:
                messages.append({"role" : previous_mess.get("author", "user"), "content": previous_mess.get("sentence", "")})

            messages.append({"role": "user", "content": message.instruction[i]})
            
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

            if self.output_style == "json":
                try:
                    responses.append(json.loads(response_text))
                except:
                    print(f"[COMMANDER] Bad model output")
                    responses.append({"think":response_text, "say":"", "action":"X"})
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
 
    