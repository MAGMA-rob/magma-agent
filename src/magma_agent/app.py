import gc
import logging
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from magma_core.protocol.agent import AgentRequest, AgentResponse

from .agents import (
    HistoryReactiveAgent,
    HistorySummaryReactiveAgent,
    TaskStateReactiveAgent,
)
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
    model_names = set()
    for model_settings in settings.models:
        if model_settings.name in model_names:
            raise ValueError(f"Duplicate model name {model_settings.name!r}.")
        model_names.add(model_settings.name)
        spec = get_model_type(model_settings.type)
        model_specs.append((model_settings, spec))

    if not model_specs:
        raise ValueError("At least one model must be declared in settings.models.")
    if not settings.agents:
        raise ValueError("At least one agent must be declared in settings.agents.")

    def clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.models = {}
        try:
            for model_settings, _spec in model_specs:
                model = load_declared_model(model_settings, settings.optimize_memory)
                model.set_prompt_log_dir(settings.prompt_log_dir)
                app.state.models[model.name] = model
            app.state.agents = {}
            for agent_settings in settings.agents:
                if agent_settings.name in app.state.agents:
                    raise ValueError(
                        f"Duplicate agent name {agent_settings.name!r}."
                    )
                try:
                    if agent_settings.type == "history_reactive":
                        commander = app.state.models[
                            agent_settings.models["commander"]
                        ]
                        if commander.model_type != "Commander":
                            raise ValueError(
                                f"Agent {agent_settings.name!r} requires a "
                                "Commander model."
                            )
                        agent = HistoryReactiveAgent(
                            agent_settings.name,
                            commander,
                        )
                    elif agent_settings.type == "history_summary_reactive":
                        summarizer = app.state.models[
                            agent_settings.models["summarizer"]
                        ]
                        commander = app.state.models[
                            agent_settings.models["commander"]
                        ]
                        if summarizer.model_type != "Summarizer":
                            raise ValueError(
                                f"Agent {agent_settings.name!r} requires a "
                                "Summarizer model."
                            )
                        if commander.model_type != "Commander":
                            raise ValueError(
                                f"Agent {agent_settings.name!r} requires a "
                                "Commander model."
                            )
                        max_context_tokens = agent_settings.options.get(
                            "max_context_tokens",
                            5000,
                        )
                        unsupported_options = sorted(
                            set(agent_settings.options) - {"max_context_tokens"}
                        )
                        if unsupported_options:
                            raise ValueError(
                                f"Unsupported history summary reactive options: "
                                f"{unsupported_options}"
                            )
                        agent = HistorySummaryReactiveAgent(
                            agent_settings.name,
                            summarizer,
                            commander,
                            max_context_tokens=max_context_tokens,
                        )
                    elif agent_settings.type == "task_state_reactive":
                        tsm = app.state.models[agent_settings.models["tsm"]]
                        dispatcher = app.state.models[
                            agent_settings.models["dispatcher"]
                        ]
                        if tsm.model_type != "TSM":
                            raise ValueError(
                                f"Agent {agent_settings.name!r} requires a TSM model."
                            )
                        if dispatcher.model_type != "Dispatcher":
                            raise ValueError(
                                f"Agent {agent_settings.name!r} requires a "
                                "Dispatcher model."
                            )
                        agent = TaskStateReactiveAgent(
                            agent_settings.name,
                            tsm,
                            dispatcher,
                        )
                    else:
                        raise ValueError(
                            f"Unknown agent type {agent_settings.type!r}."
                        )
                except KeyError as error:
                    raise ValueError(
                        f"Invalid model mapping for agent {agent_settings.name!r}: "
                        f"missing {error.args[0]!r}."
                    ) from error
                app.state.agents[agent.name] = agent
            yield
        finally:
            for model in app.state.models.values():
                del model
            app.state.models = {}
            app.state.agents = {}
            gc.collect()
            clear_cuda_cache()

    app = FastAPI(lifespan=lifespan)

    def offload(model: Any) -> None:
        model.offload()
        gc.collect()
        clear_cuda_cache()

    async def model_call(model: Any, message: Any, inference_mode: bool):
        async with model.lock:
            if settings.optimize_memory:
                model.set_device("cuda")
            try:
                return model.process_batched_entry(message, inference_mode)
            finally:
                if settings.optimize_memory:
                    offload(model)

    @app.post("/v1/responses", response_model=AgentResponse)
    async def responses(request: AgentRequest):
        try:
            agent = app.state.agents[request.agent]
        except KeyError as error:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown agent {request.agent!r}.",
            ) from error

        try:
            return await agent.process(request, model_call)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.post("/get_infos")
    async def get_infos():
        return {
            "human": False,
            "agents": [
                {
                    "name": agent_settings.name,
                    "type": agent_settings.type,
                    "models": agent_settings.models,
                }
                for agent_settings in settings.agents
            ],
            "models": [
                {
                    "name": model_settings.name,
                    "type": spec.model_type,
                    "model_id": model_settings.model_id,
                }
                for model_settings, spec in model_specs
            ],
        }

    return app
