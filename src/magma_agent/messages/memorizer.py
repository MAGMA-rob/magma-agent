from typing import Dict, List
from pydantic import BaseModel

class MessageMemorizer(BaseModel):
    memory: List
    preserved_memory_indices : List[int] = []
    think : str
    say : str
    inference_mode : bool = False

class BatchedMessageMemorizer(BaseModel):
    memory: List
    preserved_memory_indices : List = []
    think : List
    say : List