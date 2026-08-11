from typing import Any, Dict, List

from pydantic import BaseModel, Field


class MessageSummarizer(BaseModel):
    previous_summary: str = ""
    history: List[Dict[str, Any]] = Field(default_factory=list)
    inference_mode: bool = False


class BatchedMessageSummarizer(BaseModel):
    previous_summary: List[str]
    history: List[List[Dict[str, Any]]]
