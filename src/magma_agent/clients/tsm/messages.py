from typing import List

from pydantic import BaseModel


class MessageTSM(BaseModel):
    permanent_rules: List[str]
    goals: List[str]
    rules: List[str]
    todo: List[str]
    instruction: str
    inference_mode: bool = False


class BatchedMessageTSM(BaseModel):
    permanent_rules: List[List[str]]
    goals: List[List[str]]
    rules: List[List[str]]
    todo: List[List[str]]
    instruction: List[str]
