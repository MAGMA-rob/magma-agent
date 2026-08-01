import json
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


REPRESENTATION_FIELDS = ("rules", "goals", "todo")


def get_permanent_rules(memory: Dict[str, Any]) -> List[str]:
    permanent_rules = memory.get("memory_list", [])
    if permanent_rules is None:
        return []
    if not isinstance(permanent_rules, list):
        raise ValueError("Dispatcher memory['memory_list'] must be a list.")
    if any(not isinstance(rule, str) for rule in permanent_rules):
        raise ValueError(
            "Dispatcher memory['memory_list'] must contain only strings."
        )
    return permanent_rules


def format_dispatcher_history(history: List[Dict[str, Any]]) -> str:
    if not history:
        return "empty"

    lines = []
    for item in history:
        author = str(item.get("author", "UNKNOWN")).upper()
        content = item.get("content", item.get("sentence", ""))
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError:
                pass

        if author == "MODEL" and isinstance(content, dict):
            for robot, tool_call in content.items():
                if not isinstance(tool_call, dict):
                    continue
                normalized_call = {
                    "robot": robot,
                    "name": tool_call.get("name", ""),
                    "arguments": tool_call.get("arguments", {}),
                }
                lines.append(
                    "TOOL: "
                    + json.dumps(normalized_call, ensure_ascii=False)
                )
            continue

        if author == "SYSTEM":
            lines.append(
                "RESULT: " + json.dumps(content, ensure_ascii=False)
            )
            continue

        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        lines.append(f"{author}: {content}")

    return "\n".join(lines) if lines else "empty"


def get_representation_field(memory: Dict[str, Any], field_name: str) -> List[str]:
    if field_name not in memory:
        raise ValueError(f"Dispatcher memory must contain {field_name!r}.")
    value = memory[field_name]
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"Dispatcher memory[{field_name!r}] must be a list.")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(
            f"Dispatcher memory[{field_name!r}] must contain only strings."
        )
    return value


class MessageDispatcher(BaseModel):
    memory: Dict[str, Any]
    attributes: Dict[str, Any]
    history: List[Dict[str, Any]]
    function: List[Dict[str, Any]]
    inference_mode: bool = False

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: Dict[str, Any]) -> Dict[str, Any]:
        get_permanent_rules(memory)
        for field_name in REPRESENTATION_FIELDS:
            get_representation_field(memory, field_name)
        return memory


class BatchedMessageDispatcher(BaseModel):
    memory: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    history: List[List[Dict[str, Any]]]
    function: List[List[Dict[str, Any]]]

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for entry in memory:
            get_permanent_rules(entry)
            for field_name in REPRESENTATION_FIELDS:
                get_representation_field(entry, field_name)
        return memory
