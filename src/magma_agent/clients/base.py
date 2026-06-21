import asyncio
from abc import ABC, abstractmethod
from typing import Any, List


class BaseModelClient(ABC):
    model_type: str
    endpoint: str
    model_id: str
    name: str

    def __init__(
        self,
        name: str,
        model_type: str,
        model_id: str,
        endpoint: str,
    ) -> None:
        self.name = name
        self.model_type = model_type
        self.model_id = model_id
        self.endpoint = endpoint
        self.lock = asyncio.Lock()

    @abstractmethod
    def set_device(self, device: str) -> None:
        raise NotImplementedError("Must be defined in child class")

    @abstractmethod
    def offload(self) -> None:
        raise NotImplementedError("Must be defined in child class")

    @abstractmethod
    def process_batched_entry(self, message: Any, inference_mode: bool) -> List[Any]:
        raise NotImplementedError("Must be defined in child class")
