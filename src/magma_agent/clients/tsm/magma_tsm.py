from typing import List

import torch  # type: ignore

from magma_agent.messages import BatchedMessageTSM

from .base import TaskStateManager


class MagmaTSM(TaskStateManager):

    def process_batched_entry(
        self,
        message: BatchedMessageTSM,
        inference_mode: bool,
    ) -> List[str]:
        batch_size = len(message.instruction)
        if not batch_size:
            raise ValueError("BatchedMessageTSM must contain at least one instruction.")

        for field_name in ("goals", "rules", "todo"):
            field_value = getattr(message, field_name)
            if len(field_value) != batch_size:
                raise ValueError(
                    f"{field_name} must have the same length as instruction "
                    f"({len(field_value)} != {batch_size})."
                )

        formatted_inputs = []
        for i in range(batch_size):
            user_prompt = self._format_task_state(
                message.goals[i],
                message.rules[i],
                message.todo[i],
                message.instruction[i],
            )
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
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

        return [
            self.tokenizer.decode(
                generated_output[input_length:],
                skip_special_tokens=True,
            ).strip()
            for generated_output in output
        ]

    @staticmethod
    def _format_task_state(
        goals: List[str],
        rules: List[str],
        todo: List[str],
        instruction: str,
    ) -> str:
        user_prompt = "Goals:\n"
        for i, goal in enumerate(goals):
            user_prompt += f"[{i}] {goal}\n"

        user_prompt += "\nRules:\n"
        for i, rule in enumerate(rules):
            user_prompt += f"[{i}] {rule}\n"

        user_prompt += "\nToDo:\n"
        for i, task in enumerate(todo):
            user_prompt += f"[{i}] {task}\n"

        return user_prompt + f"\nQuery: {instruction}\n\n"
