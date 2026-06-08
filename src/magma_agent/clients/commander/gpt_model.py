from typing import Any, Dict, List, Optional, Sequence
import json
import os
import re

import torch

from magma_agent.messages import BatchedMessageCommander
from .base import BaseCommander
from .history import get_history_content, get_instruction_roles


from openai_harmony import (
    Role,
    Author,
    Message,
    Conversation,
    DeveloperContent,
    SystemContent,
    ReasoningEffort,
    ToolDescription,
    load_harmony_encoding,
    HarmonyEncodingName,
)



SINGLE_AGENT_INSTRUCTIONS = """You are MAGMA's single-agent robot commander.

You control a robot through optional Harmony function calls. Your job is to choose the next best response given:
- the current instruction or status update
- task attributes describing the current environment
- the full conversation history
- the available tools

Core decision rules:
- Use the history to infer what already happened, what failed, and what remains to do.
- Do not repeat a tool call that already succeeded.
- If a previous tool call failed, recover by correcting the call, choosing another valid tool, verifying state, or asking for missing information.
- Operate under partial observability: do not assume an object, location, or state exists unless it is in task attributes, memory, history, or can be checked with a tool.
- Maintain a coherent long-horizon plan, but only take the next necessary step.

Tool policy:
- Call a tool only when it is necessary for progress.
- Call at most one tool per response.
- Use only tools declared in the current tool list.
- Tool arguments must be valid JSON and must match the declared schema.
- Ground every argument in the provided context. Do not invent object names, robot names, quantities, or locations.
- Put function calls in the commentary channel using the provided Harmony tool-calling mechanism.
- Do not write tool-call JSON as plain text in the final channel.

Final-channel policy:
- Use the final channel only for the user-facing `say` text.
- If no tool is needed, answer concisely in the final channel.
- If information is missing, ask a concise clarification in the final channel instead of guessing.
- Do not expose hidden reasoning, chain of thought, or the Harmony format.
- Keep user-facing text short, operational, and consistent with any tool call.

Sometimes the user is just giving you rules and assignment, you must simply acknowledge without calling any tool. Be aware that tool execution (perception and action) are uncertain and may fail or give partial observation. Act in consequence.

Decision modes:
At each step, choose exactly one:
1. OBSERVE: gather missing or uncertain information using tools
2. ACT: execute one tool call
3. CLARIFY: ask the user for missing information

Belief tracking:
- Maintain a belief of the world based on the interaction (rules, assignment)
- If user ask to sort one or multiple object but you do not have any assignment in your history, ask for it.

Failure recovery:
If a tool fails:
- Do not retry blindly
- Consider possible causes:
  - missing object
  - wrong location
  - invalid arguments
  - execution failure
- Then:
  - verify with OBSERVE
  - try an alternative
  - or CLARIFY

Grounding constraint:
- Never invent objects, locations, or entities
- Use only information from attributes, history, or tool feedback
- If user is giving a rule "X object goes here" / "X likes Y". Just answer that you understand the rules.

Planning:
- Focus on incremental, verifiable progress
- Avoid risky or assumption-heavy actions
- Re-check the environment when in doubt

Hint:
To make coffee you must place the mug, load the right capsule and start the machine.
To wash clothes you must put each requested clothe in the machine, then put the correct detergent (if you do not know witch detergent to use you must ask to the user) then start the machine.
Be aware that you can only have one object in your gripper. If you take an object, you need to put it somewhere before taking something else.

"""


class OSSCommander(BaseCommander):

    def __init__(self, model_id: str, cpu_load: bool = False) -> None:
        self._ensure_harmony_available()
        self.encoding = self._load_harmony_encoding()
        self.stop_token_ids = self.encoding.stop_tokens_for_assistant_actions()
        self.reasoning_effort = ReasoningEffort.LOW
        super().__init__(
            model_id,
            cpu_load=False,
            dtype="auto",
            load_kwargs={"device_map": "auto"},
            runtime_device_move=False,
        )
        if cpu_load:
            print(
                "[OSS COMMANDER] optimize_memory CPU offload is disabled for gpt-oss "
                "because the model is loaded with device_map='auto'."
            )

    def process_batched_entry(self, message: BatchedMessageCommander, inference_mode: bool) -> List[Dict]:
        conversations = []
        prefill_ids = []
        instruction_roles = get_instruction_roles(message)

        self._validate_batch(message)

        for i in range(len(message.instruction)):
            tools = self._build_tool_descriptions(message.function[i])
            conversations.append(
                self._build_conversation(
                    history=message.history[i],
                    instruction=message.instruction[i],
                    instruction_role=instruction_roles[i],
                    attributes=message.attributes[i],
                    memory=message.memory[i],
                    tools=tools,
                )
            )
            prefill_ids.append(
                self.encoding.render_conversation_for_completion(conversations[-1], Role.ASSISTANT)
            )

        inputs = self._pad_prefill_ids(prefill_ids)
        generation_kwargs = self._generation_kwargs(inputs, inference_mode)
        self._debug_harmony_batch(prefill_ids, inputs, generation_kwargs)

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

    @staticmethod
    def _load_harmony_encoding() -> Any:
        try:
            return load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        except Exception as err:
            raise RuntimeError(
                "OSSCommander could not initialize the gpt-oss Harmony encoding. "
                "Make sure openai-harmony can access or cache its vocabulary before "
                "loading the model."
            ) from err

    @staticmethod
    def _validate_batch(message: BatchedMessageCommander) -> None:
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

    def _build_conversation(
        self,
        history: Sequence[Dict[str, Any]],
        instruction: str,
        instruction_role: str,
        attributes: Any,
        memory: Sequence[Any],
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

        pending_tool_recipients: List[str] = []
        for previous_message in history:
            history_messages, new_tool_recipients = self._history_to_harmony_messages(
                previous_message,
                pending_tool_recipients,
            )
            messages.extend(history_messages)
            pending_tool_recipients = new_tool_recipients

        messages.append(
            self._content_to_harmony_message(
                author=instruction_role,
                content=self._format_current_instruction(instruction, attributes, memory),
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
            parameters = _normalize_tool_parameters(parameters, tool.get("optional", []))

            tools.append(
                ToolDescription.new(
                    name,
                    tool.get("description", ""),
                    parameters=parameters,
                )
            )
        return tools

    @staticmethod
    def _format_current_instruction(instruction: str, attributes: Any, memory: Sequence[Any]) -> str:
        return (
            "Task attributes:\n"
            f"{_stringify(attributes)}\n\n"
            "Memory:\n"
            f"{_format_memory(memory)}\n\n"
            "Instruction:\n"
            f"{instruction}"
        )

    def _history_to_harmony_messages(
        self,
        previous_message: Dict[str, Any],
        pending_tool_recipients: Sequence[str],
    ) -> tuple[List[Any], List[str]]:
        author = previous_message.get("author")
        normalized_author = (author or "USER").lower()
        content = get_history_content(previous_message)

        if normalized_author in ("model", "assistant"):
            say, actions = _split_model_history_content(content)
            messages = []
            if say:
                messages.append(
                    Message.from_role_and_content(Role.ASSISTANT, say).with_channel("final")
                )
            tool_messages = _actions_to_harmony_tool_calls(actions)
            messages.extend(tool_messages)
            return messages, [message.recipient for message in tool_messages if message.recipient]

        if normalized_author in ("system", "status") and pending_tool_recipients:
            recipient = pending_tool_recipients[0]
            remaining = list(pending_tool_recipients[1:])
            return [_tool_result_message(recipient, content)], remaining

        return [
            self._content_to_harmony_message(
                author=author,
                content=content,
                current=False,
            )
        ], list(pending_tool_recipients)

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
            kwargs["eos_token_id"] = self._generation_stop_token_ids()

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
        raw_completion_ids = list(completion_ids)
        completion_ids = self._strip_stop_tokens(raw_completion_ids)
        self._debug_stripped_completion(raw_completion_ids, completion_ids)

        try:
            entries = self.encoding.parse_messages_from_completion_tokens(
                completion_ids,
                Role.ASSISTANT,
                strict=False,
            )
        except Exception as err:
            normalized_ids = self._normalized_completion_ids_for_parse(completion_ids)
            if normalized_ids != completion_ids:
                try:
                    entries = self.encoding.parse_messages_from_completion_tokens(
                        normalized_ids,
                        Role.ASSISTANT,
                        strict=False,
                    )
                except Exception as normalized_err:
                    err = normalized_err
                else:
                    return self._messages_to_commander_response(entries, sequence_mode=sequence_mode)
            return self._parse_completion_fallback(completion_ids, err, sequence_mode=sequence_mode)

        return self._messages_to_commander_response(entries, sequence_mode=sequence_mode)

    def _normalized_completion_ids_for_parse(self, completion_ids: Sequence[int]) -> List[int]:
        text = self.encoding.decode(completion_ids)
        normalized_text = _normalize_harmony_completion_text(text)
        if normalized_text == text:
            return list(completion_ids)
        try:
            return self.encoding.encode(normalized_text, allowed_special="all")
        except Exception:
            return list(completion_ids)

    def _strip_stop_tokens(self, completion_ids: Sequence[int]) -> List[int]:
        stop_token_ids = set(self._completion_terminal_token_ids())
        trimmed = list(completion_ids)
        while trimmed and trimmed[-1] in stop_token_ids:
            trimmed.pop()
        return trimmed

    def _generation_stop_token_ids(self) -> List[int]:
        return _unique_token_ids(
            list(self.stop_token_ids)
            + _token_id_list(getattr(self.tokenizer, "eos_token_id", None))
        )

    def _completion_terminal_token_ids(self) -> List[int]:
        return _unique_token_ids(
            self._generation_stop_token_ids()
            + _token_id_list(getattr(self.tokenizer, "pad_token_id", None))
        )

    def _debug_harmony_batch(
        self,
        prefill_ids: Sequence[Sequence[int]],
        inputs: Dict[str, torch.Tensor],
        generation_kwargs: Dict[str, Any],
    ) -> None:
        if not _harmony_debug_enabled():
            return

        print("[OSS COMMANDER][HARMONY DEBUG]")
        print(f"prefill_lengths={[len(ids) for ids in prefill_ids]}")
        print(f"batch_input_ids_shape={tuple(inputs['input_ids'].shape)}")
        if "attention_mask" in inputs:
            print(f"attention_mask_sums={inputs['attention_mask'].sum(dim=1).tolist()}")
        print(f"generation_eos_token_id={generation_kwargs.get('eos_token_id')}")
        print(f"completion_terminal_token_ids={self._completion_terminal_token_ids()}")
        for token_id in self._completion_terminal_token_ids():
            print(f"terminal_token[{token_id}]={self._decode_token_for_debug(token_id)!r}")

        for index, ids in enumerate(prefill_ids[:2]):
            tail = list(ids[-80:])
            print(f"prefill[{index}].tail_ids={tail}")
            print(f"prefill[{index}].tail_text={self.encoding.decode(tail)!r}")

    def _debug_harmony_completion(self, completion_ids: Sequence[int]) -> None:
        if not _harmony_debug_enabled():
            return

        ids = list(completion_ids)
        print(f"[OSS COMMANDER][HARMONY DEBUG] completion_len={len(ids)}")
        print(f"[OSS COMMANDER][HARMONY DEBUG] completion_head_ids={ids[:80]}")
        print(f"[OSS COMMANDER][HARMONY DEBUG] completion_tail_ids={ids[-80:]}")
        print(f"[OSS COMMANDER][HARMONY DEBUG] completion_text={self.encoding.decode(ids)!r}")
        try:
            tokenizer_text = self.tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as err:
            tokenizer_text = f"<tokenizer decode failed: {err}>"
        print(f"[OSS COMMANDER][HARMONY DEBUG] tokenizer_completion_text={tokenizer_text!r}")

    def _decode_token_for_debug(self, token_id: int) -> str:
        try:
            return self.encoding.decode([token_id])
        except Exception:
            pass
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=False)
        except Exception:
            return ""

    def _debug_stripped_completion(
        self,
        raw_completion_ids: Sequence[int],
        completion_ids: Sequence[int],
    ) -> None:
        if not _harmony_debug_enabled() or len(raw_completion_ids) == len(completion_ids):
            return

        stripped_ids = list(raw_completion_ids[len(completion_ids):])
        print(f"[OSS COMMANDER][HARMONY DEBUG] stripped_terminal_ids={stripped_ids}")
        for token_id in stripped_ids:
            decoded_token = self._decode_token_for_debug(token_id)
            print(f"[OSS COMMANDER][HARMONY DEBUG] stripped_token[{token_id}]={decoded_token!r}")

    @staticmethod
    def _messages_to_commander_response(entries: Sequence[Any], sequence_mode: bool) -> Dict[str, Any]:
        final_parts = []
        commentary_parts = []
        actions = []

        for entry in entries:
            entry_dict = entry.to_dict()
            channel = entry_dict.get("channel")
            recipient = entry_dict.get("recipient")
            text = _content_to_text(entry_dict.get("content", ""))

            if channel == "analysis":
                continue
            if recipient:
                actions.append(_tool_action_from_message(recipient, text))
            elif channel == "final":
                final_parts.append(text)
            elif channel == "commentary":
                commentary_parts.append(text)
            elif text:
                final_parts.append(text)

        say = "\n".join(part for part in (final_parts or commentary_parts) if part).strip()
        action: Any = actions if sequence_mode else (actions[0] if actions else {})

        return {
            "think": "",
            "say": say,
            "action": action,
        }

    def _parse_completion_fallback(
        self,
        completion_ids: Sequence[int],
        err: Exception,
        sequence_mode: bool,
    ) -> Dict[str, Any]:
        text = self.encoding.decode(completion_ids)
        say = _extract_channel_text(text, "final")
        actions = _extract_tool_actions(text)
        action: Any = actions if sequence_mode else (actions[0] if actions else {})

        if not say and not action:
            print(f"[OSS COMMANDER] Harmony parse failed: {err}")
            self._debug_harmony_completion(completion_ids)
            say = text.strip()
        elif _harmony_debug_enabled():
            print(f"[OSS COMMANDER][HARMONY DEBUG] Harmony parse failed but fallback recovered: {err}")
            self._debug_harmony_completion(completion_ids)

        return {
            "think": "",
            "say": say,
            "action": action,
        }


def _harmony_debug_enabled() -> bool:
    return False


def _token_id_list(value: Any) -> List[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        token_ids = []
        for item in value:
            token_ids.extend(_token_id_list(item))
        return token_ids
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


def _unique_token_ids(token_ids: Sequence[int]) -> List[int]:
    seen = set()
    unique = []
    for token_id in token_ids:
        if token_id in seen:
            continue
        seen.add(token_id)
        unique.append(token_id)
    return unique


def _stringify(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True)
    if value is None:
        return "{}"
    return str(value)


def _format_memory(memory: Sequence[Any]) -> str:
    if not memory:
        return "empty"
    return "\n".join(f"- {_stringify(item)}" for item in memory)


def _normalize_tool_parameters(parameters: Any, optional: Sequence[str] | None = None) -> Dict[str, Any]:
    if parameters is None:
        return {"type": "object", "properties": {}}

    if isinstance(parameters, dict) and parameters.get("type") == "object":
        schema = dict(parameters)
        schema.setdefault("properties", {})
        return schema

    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}}

    optional_names = set(optional or [])
    properties: Dict[str, Any] = {}
    required = []

    for name, spec in parameters.items():
        if not isinstance(spec, dict):
            properties[name] = {"type": _json_schema_type(spec)}
        else:
            prop = dict(spec)
            prop["type"] = _json_schema_type(prop.get("type", "string"))
            properties[name] = prop

        if name not in optional_names:
            required.append(name)

    schema: Dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def _json_schema_type(value: Any) -> str:
    if isinstance(value, type):
        value = value.__name__
    value = str(value).lower()
    return {
        "str": "string",
        "string": "string",
        "int": "integer",
        "integer": "integer",
        "float": "number",
        "double": "number",
        "number": "number",
        "bool": "boolean",
        "boolean": "boolean",
        "list": "array",
        "array": "array",
        "dict": "object",
        "object": "object",
    }.get(value, "string")


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
    arguments = _parse_json_object(text, quiet=False)
    tool_name = recipient.rsplit(".", 1)[-1]

    if arguments.get("name") and "arguments" in arguments:
        return {
            "name": arguments.get("name"),
            "arguments": arguments.get("arguments", {}),
        }

    if tool_name == recipient and "name" in arguments:
        return {
            "name": arguments.get("name"),
            "arguments": arguments.get("arguments", {}),
        }

    return {
        "name": tool_name,
        "arguments": arguments,
    }


def _split_model_history_content(content: str) -> tuple[str, List[Dict[str, Any]]]:
    parsed = _parse_json_object(content, quiet=True)
    if parsed:
        if "say" in parsed or "action" in parsed:
            return str(parsed.get("say", "") or ""), _normalize_action_list(parsed.get("action"))
        if _looks_like_action(parsed):
            return "", _normalize_action_list(parsed)

    say, action = _split_trailing_json_action(content)
    return say, _normalize_action_list(action)


def _split_trailing_json_action(content: str) -> tuple[str, Any]:
    text = content.strip()
    if not text.endswith("}"):
        return text, {}

    end = len(text)
    depth = 0
    start = None
    for idx in range(end - 1, -1, -1):
        char = text[idx]
        if char == "}":
            depth += 1
        elif char == "{":
            depth -= 1
            if depth == 0:
                start = idx
                break

    if start is None:
        return text, {}

    candidate = text[start:end]
    action = _parse_json_object(candidate, quiet=True)
    if not action or not _looks_like_action(action):
        return text, {}

    return text[:start].strip(), action


def _looks_like_action(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    if isinstance(value.get("name"), str):
        return True
    return any(isinstance(item, dict) and isinstance(item.get("name"), str) for item in value.values())


def _normalize_action_list(action: Any) -> List[Dict[str, Any]]:
    if not action:
        return []
    if isinstance(action, list):
        actions = []
        for item in action:
            actions.extend(_normalize_action_list(item))
        return actions
    if not isinstance(action, dict):
        return []
    if isinstance(action.get("name"), str):
        return [{"name": action["name"], "arguments": action.get("arguments", {}) or {}}]

    actions = []
    for value in action.values():
        if isinstance(value, dict) and isinstance(value.get("name"), str):
            actions.append({"name": value["name"], "arguments": value.get("arguments", {}) or {}})
    return actions


def _actions_to_harmony_tool_calls(actions: Sequence[Dict[str, Any]]) -> List[Any]:
    messages = []
    for action in actions:
        name = action.get("name")
        if not name:
            continue
        arguments = action.get("arguments", {}) or {}
        messages.append(
            Message.from_role_and_content(Role.ASSISTANT, json.dumps(arguments, ensure_ascii=True))
            .with_channel("commentary")
            .with_recipient(f"functions.{name}")
            .with_content_type("json")
        )
    return messages


def _tool_result_message(recipient: str, content: str) -> Any:
    return (
        Message.from_author_and_content(
            Author.new(Role.TOOL, recipient),
            _status_content_without_previous_tool_call(content),
        )
        .with_channel("commentary")
    )


def _status_content_without_previous_tool_call(content: str) -> str:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return content
    if isinstance(parsed, dict):
        parsed.pop("previous_tool_call", None)
        return json.dumps(parsed, ensure_ascii=True)
    return content


def _parse_json_object(text: str, quiet: bool = False) -> Dict[str, Any]:
    if not text.strip():
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        if not quiet:
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


def _normalize_harmony_completion_text(text: str) -> str:
    text = re.sub(
        r"(<\|start\|>assistant)(<\|channel\|>[A-Za-z0-9_-]+)\s+to=([A-Za-z0-9_.-]+)",
        r"\1 to=\3\2",
        text,
    )
    return re.sub(r"(?:<\|constrain\|>\s*)+json", "json", text)


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
