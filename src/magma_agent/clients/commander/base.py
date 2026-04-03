from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore
import torch #type: ignore
from typing import List, Dict

from magma_agent.messages import BatchedMessageCommander

class BaseCommander(ABC):

    tokenizer : AutoTokenizer
    model : AutoModelForCausalLM

    def __init__(self, model_id : str, cpu_load : bool):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
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
    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
        raise NotImplementedError("Must be defined in child class")
