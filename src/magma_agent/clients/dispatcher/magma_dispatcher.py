import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch  # type: ignore

from .base import BaseDispatcher
from .messages import (
    BatchedMessageDispatcher,
)
from .parsing import parse_dispatcher_output


class MagmaDispatcher(BaseDispatcher):

    def __init__(
        self,
        model_id: str,
        overriding_chat_template_path: Optional[str],
        cpu_load: bool,
        name: str = "dispatcher",
        max_new_tokens: int = 2500,
    ) -> None:
        super().__init__(model_id, cpu_load, name=name)
        if overriding_chat_template_path is not None:
            with open(overriding_chat_template_path, "r", encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()
        elif self.tokenizer.chat_template is None:
            default_template_path = (
                Path(__file__).resolve().parents[2]
                / "default_chat_template"
                / "dispatcher.jinja"
            )
            self.tokenizer.chat_template = default_template_path.read_text(
                encoding="utf-8"
            )

        self.max_new_tokens = max_new_tokens

    def process_batched_entry(
        self,
        message: BatchedMessageDispatcher,
        inference_mode: bool,
    ) -> List[Union[Dict[str, Any], str]]:
        formatted_inputs = []
        batch_size = len(message.mode)

        if not batch_size:
            raise ValueError(
                "BatchedMessageDispatcher must contain at least one entry."
            )

        for field_name in (
            "permanent_rules",
            "rules",
            "goals",
            "attributes",
            "history",
            "tools",
        ):
            field_value = getattr(message, field_name)
            if len(field_value) != batch_size:
                raise ValueError(
                    f"{field_name} must have the same batch length "
                    f"({len(field_value)} != {batch_size})."
                )

        for i in range(batch_size):
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [{}],
                    mode=message.mode[i],
                    tools=message.tools[i],
                    task_attributes=message.attributes[i],
                    permanent_rules=message.permanent_rules[i],
                    rules=message.rules[i],
                    goals=message.goals[i],
                    history=message.history[i],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            length = len(
                self.tokenizer(formatted_inputs[i], return_tensors="pt")["input_ids"][0]
            )
            if length == 0:
                raise ValueError(
                    "The dispatcher chat template produced an empty prompt. "
                    "Check that the loaded tokenizer/chat_template matches the "
                    "model and the MagmaDispatcher formatting arguments."
                )

        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.input_device)
        if os.getenv("MAGMA_DEBUG_TOKENIZER") == "1":
            print("[DISPATCHER][BATCH TOKENIZER DEBUG]")
            print(f"batch_input_ids_shape={tuple(inputs['input_ids'].shape)}")
            if "attention_mask" in inputs:
                print(
                    f"attention_mask_sums={inputs['attention_mask'].sum(dim=1).tolist()}"
                )

        input_lengths = [len(input_ids) for input_ids in inputs["input_ids"]]
        generation_options = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_options["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            if inference_mode:
                output = self.model.generate(
                    **inputs,
                    **generation_options,
                    do_sample=False,
                )
            else:
                output = self.model.generate(
                    **inputs,
                    **generation_options,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                )

        responses = []
        for i in range(len(formatted_inputs)):
            generated_tokens = output[i][input_lengths[i]:]
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
