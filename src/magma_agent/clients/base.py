import asyncio
from abc import ABC, abstractmethod
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Any, List
from uuid import uuid4


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
        self.prompt_log_dir: Path | None = None
        self._prompt_log_paths: ContextVar[tuple[Path, ...]] = ContextVar(
            f"prompt_log_paths_{name}_{id(self)}",
            default=(),
        )

    def set_prompt_log_dir(self, path: str | None) -> None:
        self.prompt_log_dir = Path(path) if path else None

    def log_prompt_exchange(
        self,
        prompt: str,
        response: str,
        valid: bool,
    ) -> None:
        if self.prompt_log_dir is None:
            return

        self.prompt_log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{timestamp}_{self.name}_{uuid4().hex[:8]}.log"
        )
        content = (
            "----- PROMPT -----\n"
            f"{prompt}\n"
            "----- RAW RESPONSE -----\n"
            f"{response}\n"
            "----- PARSING STATUS -----\n"
            f"{'VALID' if valid else 'INVALID'}\n"
        )
        (self.prompt_log_dir / filename).write_text(
            content,
            encoding="utf-8",
        )
        path = self.prompt_log_dir / filename
        paths = self._prompt_log_paths.get()
        self._prompt_log_paths.set((*paths, path))

    def update_prompt_log_validity(
        self,
        validities: List[bool],
    ) -> None:
        if not validities or self.prompt_log_dir is None:
            return

        paths = self._prompt_log_paths.get()
        if len(paths) < len(validities):
            return
        batch_paths = paths[-len(validities):]
        self._prompt_log_paths.set(paths[:-len(validities)])

        for path, valid in zip(batch_paths, validities):
            content = path.read_text(encoding="utf-8")
            status_header = "----- PARSING STATUS -----\n"
            prefix, separator, _ = content.partition(status_header)
            if separator:
                content = (
                    prefix
                    + "----- AGENT VALIDATION STATUS -----\n"
                    + ("VALID\n" if valid else "INVALID\n")
                )
                path.write_text(content, encoding="utf-8")

    @abstractmethod
    def set_device(self, device: str) -> None:
        raise NotImplementedError("Must be defined in child class")

    @abstractmethod
    def offload(self) -> None:
        raise NotImplementedError("Must be defined in child class")

    @abstractmethod
    def process_batched_entry(self, message: Any, inference_mode: bool) -> List[Any]:
        raise NotImplementedError("Must be defined in child class")
