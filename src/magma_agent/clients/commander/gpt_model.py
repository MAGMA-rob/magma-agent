from typing import Any, Dict, List, Optional, Sequence
import json
import re

import torch

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander
from .history import get_history_content, get_instruction_roles

try:
    from openai_harmony import (  # type: ignore
        Role,
        Message,
        Conversation,
        DeveloperContent,
        SystemContent,
        ReasoningEffort,
        ToolDescription,
        load_harmony_encoding,
        HarmonyEncodingName,
    )
except ModuleNotFoundError:
    Role = None  # type: ignore
    Message = None  # type: ignore
    Conversation = None  # type: ignore
    DeveloperContent = None  # type: ignore
    SystemContent = None  # type: ignore
    ReasoningEffort = None  # type: ignore
    ToolDescription = None  # type: ignore
    load_harmony_encoding = None  # type: ignore
    HarmonyEncodingName = None  # type: ignore


GPT_OSS_MODEL_ALIASES = {
    "gpt-oss": "openai/gpt-oss-20b",
    "gpt-oss:20b": "openai/gpt-oss-20b",
    "gpt-oss:120b": "openai/gpt-oss-120b",
    "gpt_oss": "openai/gpt-oss-20b",
    "gpt_oss:20b": "openai/gpt-oss-20b",
    "gpt_oss:120b": "openai/gpt-oss-120b",
}

SINGLE_AGENT_INSTRUCTIONS = """You are MAGMA's single-agent robot commander.

Use the full conversation history and the current task attributes to decide the next response.
There is no persistent memory agent in this mode, so do not summarize or update memory.

When a robot action is required, call exactly one available function in the commentary channel.
When no robot action is required, answer in the final channel.
Never call a function that is not declared in the current tool list.
Keep user-facing text concise and operational.
"""


class OSSCommander(BaseCommander):

    def __init__(self, model_id: str = "openai/gpt-oss-20b", cpu_load: bool = False) -> None:
        self._ensure_harmony_available()
        super().__init__(normalize_gpt_oss_model_id(model_id), cpu_load=cpu_load, dtype="auto")
        self.encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.reasoning_effort = ReasoningEffort.MEDIUM

    def process_batched_entry(self, message: BatchedMessageCommander, inference_mode: bool) -> List[Dict]:
        conversations = []
        prefill_ids = []
        instruction_roles = get_instruction_roles(message)

        for i in range(len(message.instruction)):
            tools = self._build_tool_descriptions(message.function[i])
            conversations.append(
                self._build_conversation(
                    history=message.history[i],
                    instruction=message.instruction[i],
                    instruction_role=instruction_roles[i],
                    attributes=message.attributes[i],
                    tools=tools,
                )
            )
            prefill_ids.append(
                self.encoding.render_conversation_for_completion(conversations[-1], Role.ASSISTANT)
            )

        inputs = self._pad_prefill_ids(prefill_ids)
        generation_kwargs = self._generation_kwargs(inputs, inference_mode)

        with torch.no_grad():
            output = self.model.generate(**generation_kwargs)

        prompt_length = inputs["input_ids"].shape[1]
        sequence_mode = message.prediction_mode == "sequence"
        responses = []

        for i in range(len(prefill_ids)):
            completion_ids = output[i][prompt_length:].tolist()
            responses.append(self._parse_completion(completion_ids, sequence_mode=sequence_mode))

        return responses

    @staticmethod
    def _ensure_harmony_available() -> None:
        if load_harmony_encoding is None:
            raise ModuleNotFoundError(
                "OSSCommander requires the 'openai-harmony' package. "
                "Install magma_agent dependencies or run 'pip install openai-harmony'."
            )

    def _build_conversation(
        self,
        history: Sequence[Dict[str, Any]],
        instruction: str,
        instruction_role: str,
        attributes: Any,
        tools: Sequence[Any],
    ) -> Any:
        system_content = (
            SystemContent.new()
            .with_reasoning_effort(self.reasoning_effort)
            .with_required_channels(["analysis", "commentary", "final"])
        )
        developer_content = DeveloperContent.new().with_instructions(SINGLE_AGENT_INSTRUCTIONS)
        if tools:
            developer_content = developer_content.with_function_tools(tools)

        messages = [
            Message.from_role_and_content(Role.SYSTEM, system_content),
            Message.from_role_and_content(Role.DEVELOPER, developer_content),
        ]

        for previous_message in history:
            messages.append(self._history_to_harmony_message(previous_message))

        messages.append(
            self._content_to_harmony_message(
                author=instruction_role,
                content=self._format_current_instruction(instruction, attributes),
                current=True,
            )
        )

        return Conversation.from_messages(messages)

    @staticmethod
    def _build_tool_descriptions(functions: Sequence[Dict[str, Any]]) -> List[Any]:
        tools = []
        for tool in functions:
            name = tool.get("name")
            if not name:
                raise ValueError(f"Tool declaration is missing a name: {tool}")

            parameters = tool.get("parameters", tool.get("arguments"))
            if parameters is None:
                parameters = {"type": "object", "properties": {}}

            tools.append(
                ToolDescription.new(
                    name,
                    tool.get("description", ""),
                    parameters=parameters,
                )
            )
        return tools

    @staticmethod
    def _format_current_instruction(instruction: str, attributes: Any) -> str:
        return (
            "Task attributes:\n"
            f"{_stringify(attributes)}\n\n"
            "Instruction:\n"
            f"{instruction}"
        )

    def _history_to_harmony_message(self, previous_message: Dict[str, Any]) -> Any:
        return self._content_to_harmony_message(
            author=previous_message.get("author"),
            content=get_history_content(previous_message),
            current=False,
        )

    @staticmethod
    def _content_to_harmony_message(author: Optional[str], content: str, current: bool) -> Any:
        normalized_author = (author or "USER").lower()

        if normalized_author in ("model", "assistant"):
            return Message.from_role_and_content(Role.ASSISTANT, content).with_channel("final")

        if normalized_author in ("system", "status"):
            prefix = "Current status" if current else "Status update"
            content = f"{prefix}:\n{content}"

        return Message.from_role_and_content(Role.USER, content)

    def _pad_prefill_ids(self, batched_ids: Sequence[Sequence[int]]) -> Dict[str, torch.Tensor]:
        max_length = max(len(ids) for ids in batched_ids)
        pad_token_id = self._pad_token_id()
        input_ids = []
        attention_mask = []

        for ids in batched_ids:
            pad_length = max_length - len(ids)
            input_ids.append([pad_token_id] * pad_length + list(ids))
            attention_mask.append([0] * pad_length + [1] * len(ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self.input_device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=self.input_device),
        }

    def _pad_token_id(self) -> int:
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)

        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, list) and eos_token_id:
            return int(eos_token_id[0])
        if eos_token_id is not None:
            return int(eos_token_id)

        return 0

    def _generation_kwargs(
        self,
        inputs: Dict[str, torch.Tensor],
        inference_mode: bool,
    ) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            **inputs,
            "max_new_tokens": 2500,
            "pad_token_id": self._pad_token_id(),
        }
        if self.stop_token_ids:
            kwargs["eos_token_id"] = self.stop_token_ids

        if inference_mode:
            kwargs["do_sample"] = False
        else:
            kwargs.update(
                {
                    "do_sample": True,
                    "temperature": 1.0,
                    "top_p": 1.0,
                }
            )

        return kwargs

    def _parse_completion(self, completion_ids: Sequence[int], sequence_mode: bool) -> Dict[str, Any]:
        completion_ids = self._strip_stop_tokens(completion_ids)

        try:
            entries = self.encoding.parse_messages_from_completion_tokens(
                completion_ids,
                Role.ASSISTANT,
                strict=False,
            )
        except Exception as err:
            return self._parse_completion_fallback(completion_ids, err, sequence_mode=sequence_mode)

        return self._messages_to_commander_response(entries, sequence_mode=sequence_mode)

    def _strip_stop_tokens(self, completion_ids: Sequence[int]) -> List[int]:
        stop_token_ids = set(self.stop_token_ids)
        trimmed = list(completion_ids)
        while trimmed and trimmed[-1] in stop_token_ids:
            trimmed.pop()
        return trimmed

    @staticmethod
    def _messages_to_commander_response(entries: Sequence[Any], sequence_mode: bool) -> Dict[str, Any]:
        reasoning_parts = []
        final_parts = []
        commentary_parts = []
        actions = []

        for entry in entries:
            entry_dict = entry.to_dict()
            channel = entry_dict.get("channel")
            recipient = entry_dict.get("recipient")
            text = _content_to_text(entry_dict.get("content", ""))

            if channel == "analysis":
                reasoning_parts.append(text)
            elif recipient:
                actions.append(_tool_action_from_message(recipient, text))
            elif channel == "final":
                final_parts.append(text)
            elif channel == "commentary":
                commentary_parts.append(text)
            elif text:
                final_parts.append(text)

        reasoning = "\n".join(part for part in reasoning_parts if part).strip()
        say = "\n".join(part for part in (final_parts or commentary_parts) if part).strip()
        action: Any = actions if sequence_mode else (actions[0] if actions else {})

        return {
            "reasoning": reasoning,
            "think": reasoning,
            "say": say,
            "action": action,
        }

    def _parse_completion_fallback(
        self,
        completion_ids: Sequence[int],
        err: Exception,
        sequence_mode: bool,
    ) -> Dict[str, Any]:
        print(f"[OSS COMMANDER] Harmony parse failed: {err}")
        text = self.encoding.decode(completion_ids)
        reasoning = _extract_channel_text(text, "analysis")
        say = _extract_channel_text(text, "final")
        actions = _extract_tool_actions(text)
        action: Any = actions if sequence_mode else (actions[0] if actions else {})

        if not reasoning and not say and not action:
            say = text.strip()

        return {
            "reasoning": reasoning,
            "think": reasoning,
            "say": say,
            "action": action,
        }


def normalize_gpt_oss_model_id(model_id: Optional[str]) -> str:
    if not model_id:
        return "openai/gpt-oss-20b"

    normalized = model_id.strip()
    alias_key = normalized.lower()
    if alias_key in GPT_OSS_MODEL_ALIASES:
        return GPT_OSS_MODEL_ALIASES[alias_key]

    if "/" not in normalized and re.fullmatch(r"gpt[-_]oss[-_:](20b|120b)", alias_key):
        size = re.search(r"(20b|120b)", alias_key)
        if size:
            return f"openai/gpt-oss-{size.group(1)}"

    return normalized


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return "{}"
    return str(value)


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_content_to_text(item) for item in content)
    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"])
        if "content" in content:
            return _content_to_text(content["content"])
        return json.dumps(content, ensure_ascii=True)
    if content is None:
        return ""
    return str(content)


def _tool_action_from_message(recipient: str, text: str) -> Dict[str, Any]:
    arguments = _parse_json_object(text)
    tool_name = recipient.rsplit(".", 1)[-1]

    if tool_name == recipient and "name" in arguments:
        return {
            "name": arguments.get("name"),
            "arguments": arguments.get("arguments", {}),
        }

    return {
        "name": tool_name,
        "arguments": arguments,
    }


def _parse_json_object(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        print("[OSS COMMANDER] Invalid tool call JSON")
        return {}
    if isinstance(parsed, dict):
        return parsed
    return {}


def _extract_channel_text(text: str, channel: str) -> str:
    pattern = re.compile(
        rf"<\|channel\|>{re.escape(channel)}.*?<\|message\|>(.*?)(?=<\|end\|>|<\|call\|>|<\|start\|>|$)",
        re.DOTALL,
    )
    return "\n".join(match.group(1).strip() for match in pattern.finditer(text)).strip()


def _extract_tool_actions(text: str) -> List[Dict[str, Any]]:
    pattern = re.compile(
        r"to=([A-Za-z0-9_.-]+).*?<\|message\|>(.*?)(?=<\|call\|>|<\|end\|>|<\|start\|>|$)",
        re.DOTALL,
    )
    actions = []
    for match in pattern.finditer(text):
        recipient = match.group(1)
        arguments = match.group(2).strip()
        actions.append(_tool_action_from_message(recipient, arguments))
    return actions
