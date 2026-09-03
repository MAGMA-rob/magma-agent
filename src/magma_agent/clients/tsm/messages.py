from typing import Any, List

from pydantic import BaseModel


class BatchedMessageTSM(BaseModel):
    attributes: List[dict[str, Any]]
    permanent_rules: List[List[str]]
    goals: List[List[str]]
    rules: List[List[str]]
    instruction: List[str]
