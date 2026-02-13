from typing import List, Dict
from pydantic import BaseModel

class MessageCommander(BaseModel):
    memory: List
    attributes : Dict
    history : List
    function: List
    instruction: str
    inference_mode : bool = False
    prediction_mode : str = "tool_select" #Could be tool_select or sequence

class BatchedMessageCommander(BaseModel):
    memory: List
    attributes : List
    history : List
    function: List
    instruction: List
    prediction_mode : str = "tool_select" #Could be tool_select or sequence