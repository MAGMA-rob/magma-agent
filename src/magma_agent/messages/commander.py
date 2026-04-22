from typing import List, Dict
from pydantic import BaseModel, Field

class MessageCommander(BaseModel):
    memory: List
    attributes : Dict
    history : List
    function: List
    instruction: str
    instruction_role: str = "user"
    inference_mode : bool = False
    prediction_mode : str = "tool_select" #Could be tool_select or sequence

class BatchedMessageCommander(BaseModel):
    memory: List
    attributes : List
    history : List
    function: List
    instruction: List
    instruction_role: List[str] = Field(default_factory=list)
    prediction_mode : str = "tool_select" #Could be tool_select or sequence
