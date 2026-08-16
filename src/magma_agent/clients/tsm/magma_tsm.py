import re
from typing import Dict, List, Optional, Union

import torch  # type: ignore

from .messages import BatchedMessageTSM

from .base import TaskStateManager

ParsedAction = Dict[str, Union[int, str]]


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
    ) -> List[Union[List[ParsedAction], str]]:
        batch_size = len(message.instruction)
        if not batch_size:
            raise ValueError("BatchedMessageTSM must contain at least one instruction.")

        for field_name in ("permanent_rules", "goals", "rules", "todo"):
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
                message.todo[i],
                message.instruction[i],
            )
            formatted_inputs.append(
                self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": user_prompt}],
                    permanent_rules=message.permanent_rules[i],
                    rules=message.rules[i],
                    goals=message.goals[i],
                    todo=message.todo[i],
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
            parsed_response = parse_task_state_update(response_text)
            self.log_prompt_exchange(
                formatted_inputs[index],
                response_text,
                isinstance(parsed_response, list),
            )
            responses.append(parsed_response)

        return responses

    @staticmethod
    def _format_task_state(
        permanent_rules: List[str],
        goals: List[str],
        rules: List[str],
        todo: List[str],
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
        active_goal_index = 0
        for goal in goals:
            if goal.startswith("[completed] "):
                user_prompt += f"{goal}\n"
                continue
            user_prompt += f"[{active_goal_index}] {goal}\n"
            active_goal_index += 1

        user_prompt += "\nRules:\n"
        for i, rule in enumerate(rules):
            user_prompt += f"[{i}] {rule}\n"

        user_prompt += "\nToDo:\n"
        for i, task in enumerate(todo):
            user_prompt += f"[{i}] {task}\n"

        return user_prompt + f"\nQuery: {instruction}\n\n"


def parse_task_state_update(text: str) -> Union[List[ParsedAction], str]:
    actions: List[ParsedAction] = []
    action_types = {
        "ADD_GOAL": "add_goal",
        "REMOVE_GOAL": "remove_goal",
        "ADD_RULE": "add_rule",
        "REMOVE_RULE": "remove_rule",
        "ADD_TODO": "add_todo",
        "REMOVE_TODO": "remove_todo",
    }

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        match = re.fullmatch(r"([A-Z_]+)\((.*)\)", line)
        if match is None:
            return text

        action_name = match.group(1)
        if action_name not in action_types:
            return text

        raw_content = match.group(2).strip()
        if not raw_content:
            return text

        if raw_content.lstrip("-").isdigit():
            content: Union[int, str] = int(raw_content)
        elif (
            len(raw_content) >= 2
            and raw_content[0] == raw_content[-1]
            and raw_content[0] in {"'", '"'}
        ):
            content = raw_content[1:-1]
        else:
            content = raw_content

        actions.append({"type": action_types[action_name], "content": content})

    if not actions:
        return text

    return actions
