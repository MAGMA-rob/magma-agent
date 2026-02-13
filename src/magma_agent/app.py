from fastapi import FastAPI, BackgroundTasks
from contextlib import asynccontextmanager
import logging, json
from typing import List, Dict

from .config import Settings
from .messages import BatchedMessageCommander, BatchedMessageMemorizer, MessageCommander, MessageMemorizer
from .load import load_commander, load_memorizer


def create_app(settings: Settings) -> FastAPI:

    logging.basicConfig(
        filename=settings.log_file,
        level=logging.INFO,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.commander = None
        app.state.memorizer = None

        if settings.commander_id:
            app.state.commander = load_commander(settings.commander_id, settings.optimize_memory, settings.commander_output_style, settings.commander_chat_template)

        if settings.memorizer_id:
            app.state.memorizer = load_memorizer(settings.memorizer_id, settings.optimize_memory)

        if not app.state.commander and not app.state.memorizer:
            raise ValueError("At least one agent must be provided.")

        yield

    app = FastAPI(lifespan=lifespan)

    async def offload(m):
        m.model.to("cpu")

    @app.post("/chat")
    async def chat(message: MessageCommander | BatchedMessageCommander, background_tasks: BackgroundTasks):
        if not app.state.commander:
            raise ValueError(f"No commander loaded in this instance")
        commander = app.state.commander
        if settings.optimize_memory:
            commander.model.to("cuda")

        if isinstance(message, MessageCommander):
            # Convert single message to batched format
            inf_mode = message.inference_mode
            message = BatchedMessageCommander(
                memory=[message.memory],
                attributes=[message.attributes],
                history=[message.history],
                function=[message.function],
                instruction=[message.instruction],
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
        if settings.optimize_memory:
            background_tasks.add_task(offload, commander)
        return out
        
    @app.post("/update_memory")
    async def update_memory(message: MessageMemorizer | BatchedMessageMemorizer,  background_tasks: BackgroundTasks):
        if not app.state.memorizer:
            raise ValueError(f"No memorizer loaded in this instance")
        
        memorizer = app.state.memorizer

        if settings.optimize_memory:
            memorizer.model.to("cuda")
        if isinstance(message, MessageMemorizer):
            inf_mode = message.inference_mode
            # Convert single message to batched format
            message = BatchedMessageMemorizer(
                memory=[message.memory],
                think = [message.think],
                say = [message.say],
                preserved_memory_indices = [message.preserved_memory_indices]
            )
            out = memorizer.process_batched_entry(message, inf_mode)[0]
        else:
            out = memorizer.process_batched_entry(message, False)

        if settings.optimize_memory:
            background_tasks.add_task(offload, memorizer)
        return {"update":out}

    @app.post("/get_infos")
    async def get_infos(payload : Dict):
        return {
            "commander" : settings.commander_id,
            "memorizer" : settings.memorizer_id
        }

    return app
