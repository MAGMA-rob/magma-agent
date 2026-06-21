from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator


def get_memory_list(memory: Dict[str, List[Any]]) -> List[Any]:
    return memory["memory_list"]


class MessageCommander(BaseModel):
    memory: Dict[str, List[Any]]
    attributes: Dict[str, Any]
    history: List[Dict[str, Any]]
    function: List[Dict[str, Any]]
    instruction: str
    instruction_role: str = "USER"
    inference_mode: bool = False
    prediction_mode: str = "tool_select"

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        if "memory_list" not in memory:
            raise ValueError("Commander memory must contain a memory_list key.")
        return memory


class BatchedMessageCommander(BaseModel):
    memory: List[Dict[str, List[Any]]]
    attributes: List[Dict[str, Any]]
    history: List[List[Dict[str, Any]]]
    function: List[List[Dict[str, Any]]]
    instruction: List[str]
    instruction_role: List[str] = Field(default_factory=list)
    prediction_mode: str = "tool_select"

    @field_validator("memory")
    @classmethod
    def validate_memory(cls, memory: List[Dict[str, List[Any]]]) -> List[Dict[str, List[Any]]]:
        for index, item in enumerate(memory):
            if "memory_list" not in item:
                raise ValueError(f"Commander batch memory[{index}] must contain a memory_list key.")
        return memory
