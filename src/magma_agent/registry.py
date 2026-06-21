import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel

from .clients.base import BaseModelClient
from .clients.commander.loader import load_commander
from .clients.commander.messages import BatchedMessageCommander, MessageCommander
from .clients.tsm.loader import load_tsm
from .clients.tsm.messages import BatchedMessageTSM, MessageTSM
from .config import ModelSettings


@dataclass(frozen=True)
class ModelTypeSpec:
    model_type: str
    default_endpoint: str
    single_message: Type[BaseModel]
    batched_message: Type[BaseModel]
    load: Callable[[str, str, str, bool, Dict[str, Any]], BaseModelClient]
    single_to_batch: Callable[[BaseModel], tuple[BaseModel, bool]]
    format_single_response: Callable[[Any], Any]
    format_batch_response: Callable[[BaseModel, List[Any]], Any]


def commander_single_to_batch(message: BaseModel) -> tuple[BaseModel, bool]:
    commander_message = message
    return (
        BatchedMessageCommander(
            memory=[commander_message.memory],
            attributes=[commander_message.attributes],
            history=[commander_message.history],
            function=[commander_message.function],
            instruction=[commander_message.instruction],
            instruction_role=[commander_message.instruction_role],
            prediction_mode=commander_message.prediction_mode,
        ),
        commander_message.inference_mode,
    )


def format_commander_single_response(answer: Any) -> Any:
    return answer


def format_commander_batch_response(message: BaseModel, answers: List[Any]) -> Dict[str, List[Any]]:
    out: Dict[str, List[Any]] = {
        "think": [],
        "say": [],
        "action": [],
    }

    for answer in answers:
        if isinstance(answer, str):
            answer = json.loads(answer)

        out["think"].append(answer.get("think", ""))
        out["say"].append(answer.get("say", ""))

        if message.prediction_mode == "sequence":
            out["action"].append(answer.get("action", []))
            continue

        action = answer.get("action", {})
        if isinstance(action, list):
            print(
                "[COMMANDER] Model returned a sequence but prediction_mode is set to tool_select"
            )
            action = action[0] if action else {}
        out["action"].append(action)

    return out


def tsm_single_to_batch(message: BaseModel) -> tuple[BaseModel, bool]:
    tsm_message = message
    return (
        BatchedMessageTSM(
            goals=[tsm_message.goals],
            rules=[tsm_message.rules],
            todo=[tsm_message.todo],
            instruction=[tsm_message.instruction],
        ),
        tsm_message.inference_mode,
    )


def format_tsm_single_response(answer: Any) -> Dict[str, Any]:
    return {"update": answer}


def format_tsm_batch_response(message: BaseModel, answers: List[Any]) -> Dict[str, Any]:
    return {"update": answers}


MODEL_TYPES: Dict[str, ModelTypeSpec] = {
    "Commander": ModelTypeSpec(
        model_type="Commander",
        default_endpoint="/chat",
        single_message=MessageCommander,
        batched_message=BatchedMessageCommander,
        load=load_commander,
        single_to_batch=commander_single_to_batch,
        format_single_response=format_commander_single_response,
        format_batch_response=format_commander_batch_response,
    ),
    "TSM": ModelTypeSpec(
        model_type="TSM",
        default_endpoint="/update_task_state",
        single_message=MessageTSM,
        batched_message=BatchedMessageTSM,
        load=load_tsm,
        single_to_batch=tsm_single_to_batch,
        format_single_response=format_tsm_single_response,
        format_batch_response=format_tsm_batch_response,
    ),
}


def get_model_type(model_type: str) -> ModelTypeSpec:
    try:
        return MODEL_TYPES[model_type]
    except KeyError as err:
        available = ", ".join(sorted(MODEL_TYPES))
        raise ValueError(f"Unknown model type {model_type!r}. Available types: {available}") from err


def load_declared_model(settings: ModelSettings, optimize_memory: bool) -> BaseModelClient:
    spec = get_model_type(settings.type)
    endpoint = settings.endpoint or spec.default_endpoint
    return spec.load(
        settings.name,
        settings.model_id,
        endpoint,
        optimize_memory,
        settings.options,
    )
