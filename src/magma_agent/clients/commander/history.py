from typing import Any, Dict, List
import json

from magma_agent.messages import BatchedMessageCommander


def _stringify_content(content: Any) -> str:
    if isinstance(content, (dict, list)):
        return json.dumps(content, ensure_ascii=True)
    if content is None:
        return ""
    return str(content)


def get_history_content(message: Dict[str, Any]) -> str:
    return _stringify_content(message.get("content", message.get("sentence", "")))


def get_instruction_roles(message: BatchedMessageCommander) -> List[str]:
    roles = getattr(message, "instruction_role", [])
    if not roles:
        return ["user"] * len(message.instruction)
    if len(roles) != len(message.instruction):
        raise ValueError(
            "instruction_role must have the same length as instruction in BatchedMessageCommander"
        )
    return roles


def map_chat_role(
    author: str | None,
    system_role: str = "system",
    model_role: str = "assistant",
) -> str:
    if author in ("model", "assistant"):
        return model_role
    if author in ("system", "status"):
        return system_role
    if author in (None, ""):
        return "user"
    return str(author)
