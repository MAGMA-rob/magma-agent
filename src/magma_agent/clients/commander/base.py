from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore
import torch #type: ignore
from typing import List, Dict

from magma_agent.messages import BatchedMessageCommander

class BaseCommander(ABC):

    tokenizer : AutoTokenizer
    model : AutoModelForCausalLM

    def __init__(self, model_id : str, cpu_load : bool):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu" if cpu_load else "cuda", dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model.eval()

    @abstractmethod
    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
        raise NotImplementedError("Must be defined in child class")