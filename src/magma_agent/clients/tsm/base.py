from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch
from abc import ABC, abstractmethod

from magma_agent.clients.base import BaseModelClient
from .messages import BatchedMessageTSM


class TaskStateManager(BaseModelClient, ABC):

    def __init__(
        self,
        model_id,
        cpu_load: bool,
        tokenizer=None,
        name: str = "tsm",
        endpoint: str = "/update_task_state",
    ) -> None:
        super().__init__(
            name=name,
            model_type="TSM",
            model_id=model_id,
            endpoint=endpoint,
        )
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                padding_side="left",
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._current_device = "cpu"
        self.set_device("cpu" if cpu_load else "cuda")

    def set_device(self, device: str) -> None:
        if self._current_device == device:
            return
        self.model.to(device)
        self._current_device = device

    def offload(self) -> None:
        self.set_device("cpu")

    @abstractmethod
    def process_batched_entry(
        self,
        message: BatchedMessageTSM,
        inference_mode: bool,
    ) -> List[str]:
        raise NotImplementedError("Must be defined in child class")
