from fastapi import FastAPI
from contextlib import asynccontextmanager
import asyncio
import gc
import logging, json
from typing import List, Dict
import torch

from .config import Settings
from .messages import BatchedMessageCommander, BatchedMessageTSM, MessageCommander, MessageTSM
from .load import load_commander, load_tsm


def create_app(settings: Settings) -> FastAPI:

    logging.basicConfig(
        filename=settings.log_file,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    def clear_cuda_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.commander = None
        app.state.tsm = None
        app.state.commander_lock = asyncio.Lock()
        app.state.tsm_lock = asyncio.Lock()

        try:
            if settings.commander_id:
                app.state.commander = load_commander(
                    settings.commander_id,
                    settings.optimize_memory,
                    settings.commander_output_style,
                    settings.commander_chat_template,
                    qwen_options={
                        "quantization_mode": settings.qwen_quantization,
                        "max_new_tokens": settings.qwen_max_new_tokens,
                        "attn_implementation": settings.qwen_attn_implementation,
                        "use_cache": settings.qwen_use_cache,
                        "enable_thinking": settings.qwen_enable_thinking,
                        "device_map": settings.qwen_device_map,
                        "gpu_memory_limit": settings.qwen_gpu_memory_limit,
                        "allow_cpu_offload": settings.qwen_allow_cpu_offload,
                        "offload_folder": settings.qwen_offload_folder,
                    },
                )

            if settings.tsm_id:
                app.state.tsm = load_tsm(settings.tsm_id, settings.optimize_memory)

            if not app.state.commander and not app.state.tsm:
                raise ValueError("At least one agent must be provided.")

            yield
        finally:
            if app.state.commander is not None:
                del app.state.commander
                app.state.commander = None
            if app.state.tsm is not None:
                del app.state.tsm
                app.state.tsm = None
            gc.collect()
            clear_cuda_cache()

    app = FastAPI(lifespan=lifespan)

    def offload(m):
        m.offload()
        gc.collect()
        clear_cuda_cache()

    @app.post("/chat")
    async def chat(message: MessageCommander | BatchedMessageCommander):
        if not app.state.commander:
            raise ValueError(f"No commander loaded in this instance")
        commander = app.state.commander

        async with app.state.commander_lock:
            if settings.optimize_memory:
                commander.set_device("cuda")

            try:
                if isinstance(message, MessageCommander):
                    # Convert single message to batched format
                    inf_mode = message.inference_mode
                    message = BatchedMessageCommander(
                        memory=[message.memory],
                        attributes=[message.attributes],
                        history=[message.history],
                        function=[message.function],
                        instruction=[message.instruction],
                        instruction_role=[message.instruction_role],
                        prediction_mode = message.prediction_mode
                    )
                    out = commander.process_batched_entry(message, inf_mode)[0]
                else:
                    answers = commander.process_batched_entry(message, False)
                    out = {
                        "think" : [],
                        "say" : [],
                        "action": []
                    }

                    for answer in answers:
                        if isinstance(answer, str):
                            answer = json.loads(answer)
                        out["think"].append(answer.get("think", ""))
                        out["say"].append(answer.get("say", ""))
                        if message.prediction_mode == "sequence":
                            out["action"].append(answer.get("action", []))
                        else:
                            ac = answer.get("action", {})
                            if isinstance(ac, List) and not message.prediction_mode == "sequence":
                                print("[COMMANDER] Model return a sequence but the prediction_mode in the payload is set to tool_select")
                                ac = ac[0] if ac else {}
                            out["action"].append(ac)
            finally:
                if settings.optimize_memory:
                    offload(commander)
        return out
        
    @app.post("/update_task_state")
    async def update_task_state(message: MessageTSM | BatchedMessageTSM):
        if not app.state.tsm:
            raise ValueError(f"No TSM loaded in this instance")
        
        tsm = app.state.tsm

        async with app.state.tsm_lock:
            if settings.optimize_memory:
                tsm.set_device("cuda")
            try:
                if isinstance(message, MessageTSM):
                    inf_mode = message.inference_mode
                    # Convert single message to batched format
                    message = BatchedMessageTSM(
                        goals=[message.goals],
                        rules=[message.rules],
                        todo=[message.todo],
                        instruction=[message.instruction],
                    )
                    out = tsm.process_batched_entry(message, inf_mode)[0]
                else:
                    out = tsm.process_batched_entry(message, False)
            finally:
                if settings.optimize_memory:
                    offload(tsm)

        return {"update":out}

    @app.post("/get_infos")
    async def get_infos(payload : Dict):
        return {
            "commander" : settings.commander_id,
            "tsm" : settings.tsm_id
        }

    return app
