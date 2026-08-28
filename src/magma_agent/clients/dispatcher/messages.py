from typing import Any, List, Literal

from pydantic import BaseModel


class BatchedMessageDispatcher(BaseModel):
    mode: List[Literal["execution", "execution_report"]]
    permanent_rules: List[List[str]]
    rules: List[List[str]]
    goals: List[List[str]]
    attributes: List[dict[str, Any]]
    history: List[str]
    tools: List[List[dict[str, Any]]]
