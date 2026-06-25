from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


REPRESENTATION_FIELDS = ("rules", "goals", "todo")


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
    instruction: str
    inference_mode: bool = False
    prediction_mode: str = "tool_select"

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: Dict[str, Any]) -> Dict[str, Any]:
        for field_name in REPRESENTATION_FIELDS:
            get_representation_field(memory, field_name)
        return memory


class BatchedMessageDispatcher(BaseModel):
    memory: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    history: List[List[Dict[str, Any]]]
    function: List[List[Dict[str, Any]]]
    instruction: List[str]
    prediction_mode: str = "tool_select"

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for entry in memory:
            for field_name in REPRESENTATION_FIELDS:
                get_representation_field(entry, field_name)
        return memory
