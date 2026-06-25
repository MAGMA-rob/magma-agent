from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import torch  # type: ignore

from magma_agent.clients.commander.base import BaseCommander

from .messages import BatchedMessageDispatcher


class BaseDispatcher(BaseCommander, ABC):

    def __init__(
        self,
        model_id: str,
        cpu_load: bool,
        name: str = "dispatcher",
        endpoint: str = "/dispatch",
        dtype: Any = torch.float16,
        quantization=None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        runtime_device_move: bool = True,
    ) -> None:
        super().__init__(
            model_id=model_id,
            cpu_load=cpu_load,
            name=name,
            endpoint=endpoint,
            dtype=dtype,
            quantization=quantization,
            load_kwargs=load_kwargs,
            runtime_device_move=runtime_device_move,
        )
        self.model_type = "Dispatcher"

    @abstractmethod
    def process_batched_entry(
        self,
        message: BatchedMessageDispatcher,
        inference_mode: bool,
    ) -> List[Any]:
        raise NotImplementedError("Must be defined in child class")
