from typing import Dict, List
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore
import torch
from abc import ABC, abstractmethod

from magma_agent.messages import BatchedMessageMemorizer

system_prompt = "You must update a memory based on different inputs. This memory must represent past orders, important element that another model should access to simulate an external memory table for its all life.\nTo do this, you have access to :\n- The current memory\n- the think process of a precedent model\n- the answer of a precedent model\n\nThe memory will be presented as follow :\nMemory :\n[ID] I need to remember ...\n\nIf ID = X, you cannot modify this statement. Typically, you should not modify, default attribution and base task description. You can add new statement instead to override them.\n\nTo modify the memory, you can interact only using these tools :\n[{\"name\" : \"add\", \"description\":\"Add multiple new statements to the memory\", \"arguments\":{\"statements\":{\"description\":\"A list of statement to add to the memory, \"type\":List}}},{\"name\" : \"delete\", \"description\":\"Remove multiple statements from the memory which are not longer usefull\", \"arguments\":{\"ids\":{\"description\": \"The list of ids corresponding to statement you want to delete\", \"type\":List}}}]\n"

class Memorizer(ABC):

    def __init__(self, model_id, cpu_load : bool, tokenizer = None) -> None:
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cpu" if cpu_load else "cuda")
        self.model.eval()

    @abstractmethod
    def process_batched_entry(self, message : BatchedMessageMemorizer, inference_mode : bool) -> List[Dict]:
        raise NotImplementedError("Must be defined in child class")