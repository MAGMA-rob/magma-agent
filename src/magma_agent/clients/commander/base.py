from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM #type: ignore
import torch #type: ignore
from typing import Any, List, Dict, Optional

from magma_agent.clients.base import BaseModelClient
from .messages import BatchedMessageCommander


class BaseCommander(BaseModelClient, ABC):

    tokenizer : AutoTokenizer
    model : AutoModelForCausalLM

    def __init__(
        self,
        model_id : str,
        cpu_load : bool,
        name: str = "commander",
        endpoint: str = "/chat",
        dtype: Any = torch.float16,
        quantization = None,
        load_kwargs: Optional[Dict[str, Any]] = None,
        runtime_device_move: bool = True,
    ):
        super().__init__(
            name=name,
            model_type="Commander",
            model_id=model_id,
            endpoint=endpoint,
        )
        model_kwargs = {
            "dtype": dtype,
            "low_cpu_mem_usage": True,
        }
        if quantization:
            model_kwargs["quantization_config"] = quantization

        if load_kwargs:
            model_kwargs.update(load_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.eval()
        self._runtime_device_move = runtime_device_move
        self._fixed_device_warning_shown = False
        self._current_device = str(self.input_device)
        if self._runtime_device_move:
            self.set_device("cpu" if cpu_load else "cuda")

    @property
    def input_device(self) -> torch.device:
        try:
            return self.model.get_input_embeddings().weight.device
        except Exception:
            pass

        for param in self.model.parameters():
            return param.device

        return torch.device("cpu")

    def set_device(self, device: str) -> None:
        target = torch.device(device)
        current = self.input_device

        if current.type == target.type and (target.index is None or current.index == target.index):
            self._current_device = str(current)
            return

        if not self._runtime_device_move:
            if not self._fixed_device_warning_shown:
                print(
                    "[COMMANDER] Runtime CPU/GPU moves are disabled for this model "
                    f"(loaded on {current}). Keeping it in place."
                )
                self._fixed_device_warning_shown = True
            self._current_device = str(current)
            return

        self.model.to(device)
        self._current_device = str(self.input_device)

    def offload(self) -> None:
        self.set_device("cpu")

    @abstractmethod
    def process_batched_entry(self, message : BatchedMessageCommander, inference_mode : bool) -> List[Dict]:
        raise NotImplementedError("Must be defined in child class")
