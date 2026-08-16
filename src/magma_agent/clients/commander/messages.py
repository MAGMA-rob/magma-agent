from typing import Any, Dict, List

from pydantic import BaseModel, Field


def get_memory_list(memory: Dict[str, Any]) -> List[Any]:
    memory_list = memory.get("memory_list", [])
    if memory_list is None:
        return []
    if not isinstance(memory_list, list):
        raise ValueError("Commander memory['memory_list'] must be a list when provided.")
    return memory_list


class BatchedMessageCommander(BaseModel):
    memory: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    history: List[List[Dict[str, Any]]]
    function: List[List[Dict[str, Any]]]
    instruction: List[str]
    instruction_role: List[str] = Field(default_factory=list)
    prediction_mode: str = "tool_select"
