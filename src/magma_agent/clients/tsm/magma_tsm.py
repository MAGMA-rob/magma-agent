from typing import List, Optional

import torch  # type: ignore

from .messages import BatchedMessageTSM

from .base import TaskStateManager

class MagmaTSM(TaskStateManager):

    def __init__(
        self,
        model_id,
        cpu_load: bool,
        overriding_chat_template_path: Optional[str] = None,
        name: str = "tsm",
    ) -> None:
        super().__init__(model_id, cpu_load, name=name)
        if overriding_chat_template_path is not None:
            with open(overriding_chat_template_path, "r", encoding="utf-8") as f:
                self.tokenizer.chat_template = f.read()

    def process_batched_entry(
        self,
        message: BatchedMessageTSM,
        inference_mode: bool,
    ) -> List[str]:
        batch_size = len(message.instruction)
        if not batch_size:
            raise ValueError("BatchedMessageTSM must contain at least one instruction.")

        for field_name in ("permanent_rules", "goals", "rules"):
            field_value = getattr(message, field_name)
            if len(field_value) != batch_size:
                raise ValueError(
                    f"{field_name} must have the same length as instruction "
                    f"({len(field_value)} != {batch_size})."
                )

        formatted_inputs = []
        for i in range(batch_size):
            user_prompt = self._format_task_state(
                message.permanent_rules[i],
                message.goals[i],
                message.rules[i],
                message.instruction[i],
            )
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    permanent_rules=message.permanent_rules[i],
                    rules=message.rules[i],
                    goals=message.goals[i],
                    instruction=message.instruction[i],
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
            "max_new_tokens": 1024,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if inference_mode:
            generation_options["do_sample"] = False
        else:
            generation_options.update(
                {
                    "do_sample": True,
                    "temperature": 0.8,
                    "top_p": 0.95,
                }
            )

        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_options)

        responses = []
        for index, generated_output in enumerate(output):
            response_text = self.tokenizer.decode(
                generated_output[input_length:],
                skip_special_tokens=True,
            ).strip()
            self.log_prompt_exchange(
                formatted_inputs[index],
                response_text,
                True,
            )
            responses.append(response_text)

        return responses

    @staticmethod
    def _format_task_state(
        permanent_rules: List[str],
        goals: List[str],
        rules: List[str],
        instruction: str,
    ) -> str:
        user_prompt = (
            "Permanent rules (immutable guidance; never add, remove, or modify):\n"
        )
        for rule in permanent_rules:
            user_prompt += f"- {rule}\n"
        if not permanent_rules:
            user_prompt += "empty\n"

        user_prompt += "\nGoals:\n"
        for goal in goals:
            user_prompt += f"{goal}\n"
        if not goals:
            user_prompt += "empty\n"

        user_prompt += "\nRules:\n"
        for rule in rules:
            user_prompt += f"{rule}\n"
        if not rules:
            user_prompt += "empty\n"

        return user_prompt + f"\nQuery: {instruction}\n\n"
