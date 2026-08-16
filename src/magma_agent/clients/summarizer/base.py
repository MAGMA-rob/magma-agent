from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from magma_agent.clients.base import BaseModelClient
from magma_agent.clients.commander.history import (
    format_history_content,
    map_chat_role,
)

from .messages import BatchedMessageSummarizer


class MagmaSummarizer(BaseModelClient):
    def __init__(
        self,
        model_id: str,
        cpu_load: bool,
        max_new_tokens: int,
        name: str = "summarizer",
    ) -> None:
        super().__init__(
            name=name,
            model_type="Summarizer",
            model_id=model_id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.chat_template is None:
            raise ValueError(
                f"Summarizer model {model_id!r} must provide a native chat template."
            )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self._current_device = "cpu"
        self.set_device("cpu" if cpu_load else "cuda")

    def set_device(self, device: str) -> None:
        if self._current_device == device:
            return
        self.model.to(device)
        self._current_device = device

    def offload(self) -> None:
        self.set_device("cpu")

    def process_batched_entry(
        self,
        message: BatchedMessageSummarizer,
        inference_mode: bool,
    ) -> List[str]:
        if len(message.previous_summary) != len(message.history):
            raise ValueError(
                "Summarizer previous_summary and history must have the same length."
            )
        if not message.history:
            return []

        system_prompt = (
            "You maintain a compact memory of a long-horizon robotic interaction.\n"
            "Preserve information that may affect future decisions.\n"
            "Discard obsolete and irrelevant details.\n"
            "Return only the updated summary."
        )
        formatted_inputs = []
        for previous_summary, history in zip(
            message.previous_summary,
            message.history,
        ):
            interaction = []
            for history_message in history:
                role = map_chat_role(history_message.get("author"))
                interaction.append(
                    f"{role.upper()}:\n"
                    f"{format_history_content(history_message, role)}"
                )
            recent_interaction = "\n\n".join(interaction)
            user_prompt = (
                "Previous summary:\n"
                f"{previous_summary or 'empty'}\n\n"
                "Recent interaction:\n"
                f"{recent_interaction}"
            )
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            formatted_inputs,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)
        input_length = inputs["input_ids"].shape[1]
        generation_options = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.tokenizer.eos_token_id is not None:
            generation_options["eos_token_id"] = self.tokenizer.eos_token_id
        if inference_mode:
            generation_options["do_sample"] = False
        else:
            generation_options.update({
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            })

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **generation_options)

        summaries = []
        for index, generated_output in enumerate(outputs):
            summary = self.tokenizer.decode(
                generated_output[input_length:],
                skip_special_tokens=True,
            ).strip()
            self.log_prompt_exchange(
                formatted_inputs[index],
                summary,
                bool(summary),
            )
            summaries.append(summary)
        return summaries
