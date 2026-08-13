from typing import Any, Dict, List

from pydantic import BaseModel


class BatchedMessageSummarizer(BaseModel):
    previous_summary: List[str]
    history: List[List[Dict[str, Any]]]
