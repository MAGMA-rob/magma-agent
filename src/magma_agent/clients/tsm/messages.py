from typing import List

from pydantic import BaseModel


class BatchedMessageTSM(BaseModel):
    permanent_rules: List[List[str]]
    goals: List[List[str]]
    rules: List[List[str]]
    instruction: List[str]
