from typing import Optional

from .clients.commander import MagmaCommander, SmolLMCommander, QwenCommander
from .clients.memorizer import MagmaMemorizer

def load_commander(commander_id : str, optimize_memory : bool, commander_output : str, chat_template : Optional[str]=None):
    if commander_id != "none":
        if "original-qwen" in commander_id:
            print(f"[MAGMA AGENT] Loading QWEN Commander Model : qwen3-{commander_id}")
            return QwenCommander(commander_id, cpu_load=optimize_memory)
        elif "original-smollm" in commander_id:
            print("[MAGMA AGENT] Loading SmolLM3")
            return SmolLMCommander(commander_id, optimize_memory)
        else:
            print(f"[MAGMA AGENT] Loading Commander Model : {commander_id} with output format {commander_output}")
            return MagmaCommander(cpu_load=optimize_memory, model_id=commander_id, output_style=commander_output, overriding_chat_template_path=chat_template)
            
def load_memorizer(memorizer_id : str, optimize_memory : bool):
    if memorizer_id != "none":
        print(f"[MAGMA AGENT] Loading Memorizer Model : {memorizer_id}")
        return MagmaMemorizer(cpu_load=optimize_memory, model_id=memorizer_id)

    return None
