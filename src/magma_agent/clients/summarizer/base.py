import json
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from magma_agent.clients.base import BaseModelClient
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

        system_prompt = """
You are a context summarization model for a long-horizon robotic agent.

Compress the interaction history into a compact memory that allows another
agent to continue the mission without access to the discarded context.

Discard redundant dialogue, obsolete information, intermediate reasoning
that no longer matters, and details that cannot influence future decisions.

When information has been updated or contradicted, retain the latest valid
information.
"""
        formatted_inputs = []
        for previous_summary, history in zip(
            message.previous_summary,
            message.history,
        ):
            history_lines = []
            for history_message in history:
                author = str(
                    history_message.get("author", "UNKNOWN")
                ).upper()
                content: Any = history_message.get(
                    "content",
                    history_message.get("sentence", ""),
                )
                if isinstance(content, str):
                    try:
                        content = json.loads(content)
                    except json.JSONDecodeError:
                        pass

                if author == "MODEL" and isinstance(content, dict):
                    for robot, tool_call in content.items():
                        if not isinstance(tool_call, dict):
                            continue
                        normalized_call: Dict[str, Any] = {
                            "robot": robot,
                            "name": tool_call.get("name", ""),
                            "arguments": tool_call.get("arguments", {}),
                        }
                        history_lines.append(
                            "TOOL: "
                            + json.dumps(
                                normalized_call,
                                ensure_ascii=False,
                            )
                        )
                    continue

                if author == "SYSTEM":
                    history_lines.append(
                        "RESULT: "
                        + json.dumps(content, ensure_ascii=False)
                    )
                    continue

                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False)
                history_lines.append(f"{author}: {content}")

            formatted_history = (
                "\n".join(history_lines) if history_lines else "empty"
            )
            if previous_summary == "":
                previous_summary = "None"
            user_prompt = (
                f"Previous Summary:\n{previous_summary}\n\n"
                "Interactions since previous summary:\n"
                f"{formatted_history}"
            )
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    tokenize=False,
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
