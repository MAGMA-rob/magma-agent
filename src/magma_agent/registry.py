from dataclasses import dataclass
from typing import Any, Callable, Dict

from .clients.base import BaseModelClient
from .clients.commander.loader import load_commander
from .clients.dispatcher.loader import load_dispatcher
from .clients.summarizer.loader import load_summarizer
from .clients.tsm.loader import load_tsm
from .config import ModelSettings


ModelLoader = Callable[
    [str, str, bool, Dict[str, Any]],
    BaseModelClient,
]


@dataclass(frozen=True)
class ModelTypeSpec:
    model_type: str
    load: ModelLoader


MODEL_TYPES: Dict[str, ModelTypeSpec] = {
    "Commander": ModelTypeSpec("Commander", load_commander),
    "TSM": ModelTypeSpec("TSM", load_tsm),
    "Dispatcher": ModelTypeSpec("Dispatcher", load_dispatcher),
    "Summarizer": ModelTypeSpec("Summarizer", load_summarizer),
}


def get_model_type(model_type: str) -> ModelTypeSpec:
    try:
        return MODEL_TYPES[model_type]
    except KeyError as error:
        available = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(
            f"Unknown model type {model_type!r}. Available types: {available}"
        ) from error


def load_declared_model(
    settings: ModelSettings,
    optimize_memory: bool,
) -> BaseModelClient:
    return get_model_type(settings.type).load(
        settings.name,
        settings.model_id,
        optimize_memory,
        settings.options,
    )
