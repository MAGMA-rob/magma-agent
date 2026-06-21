import gc
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException
from pydantic import ValidationError

from .config import Settings
from .registry import get_model_type, load_declared_model


def create_app(settings: Settings) -> FastAPI:
    logging.basicConfig(
        filename=settings.log_file,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    model_specs = []
    endpoint_to_name: Dict[str, str] = {}
    for model_settings in settings.models:
        spec = get_model_type(model_settings.type)
        endpoint = model_settings.endpoint or spec.default_endpoint
        if endpoint in endpoint_to_name:
            raise ValueError(
                f"Endpoint {endpoint!r} is declared by both "
                f"{endpoint_to_name[endpoint]!r} and {model_settings.name!r}."
            )
        endpoint_to_name[endpoint] = model_settings.name
        model_specs.append((model_settings, spec, endpoint))

    if not model_specs:
        raise ValueError("At least one model must be declared in settings.models.")

    def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = {}
        try:
            for model_settings, _spec, _endpoint in model_specs:
                model = load_declared_model(model_settings, settings.optimize_memory)
                app.state.models[model.name] = model
            yield
        finally:
            for model in app.state.models.values():
                del model
            app.state.models = {}
            gc.collect()
            clear_cuda_cache()

    app = FastAPI(lifespan=lifespan)

    def offload(model: Any) -> None:
        model.offload()
        gc.collect()
        clear_cuda_cache()

    def parse_payload(spec: Any, payload: Dict[str, Any]) -> tuple[bool, Any, Any]:
        try:
            single_message = spec.single_message.model_validate(payload)
            batched_message, inference_mode = spec.single_to_batch(single_message)
            return True, batched_message, inference_mode
        except ValidationError as single_error:
            try:
                return False, spec.batched_message.model_validate(payload), False
            except ValidationError as batch_error:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Payload does not match single or batched message schema. "
                        f"Single error: {single_error}. Batch error: {batch_error}."
                    ),
                ) from batch_error

    for model_settings, spec, endpoint in model_specs:

        async def model_endpoint(
            payload: Dict[str, Any],
            model_name: str = model_settings.name,
            model_spec: Any = spec,
        ):
            model = app.state.models[model_name]
            is_single, message, inference_mode = parse_payload(model_spec, payload)

            async with model.lock:
                if settings.optimize_memory:
                    model.set_device("cuda")

                try:
                    answers = model.process_batched_entry(message, inference_mode)
                finally:
                    if settings.optimize_memory:
                        offload(model)

            if is_single:
                return model_spec.format_single_response(answers[0])
            return model_spec.format_batch_response(message, answers)

        model_endpoint.__name__ = f"{model_settings.name}_endpoint"
        app.post(endpoint)(model_endpoint)

    @app.post("/get_infos")
    async def get_infos(payload: Dict[str, Any]):
        return {
            "models": [
                {
                    "name": model_settings.name,
                    "type": spec.model_type,
                    "endpoint": endpoint,
                    "model_id": model_settings.model_id,
                }
                for model_settings, spec, endpoint in model_specs
            ]
        }

    return app
