from typing import Any, Dict, Optional

from .clients.commander import MagmaCommander, SmolLMCommander, QwenCommander, OSSCommander
from .clients.tsm import MagmaTSM

def load_commander(
    commander_id : str,
    optimize_memory : bool,
    commander_output : str,
    chat_template : Optional[str]=None,
    qwen_options: Optional[Dict[str, Any]] = None,
):
    if commander_id != "none":
        normalized_commander_id = commander_id.lower()
        if "qwen" in normalized_commander_id:
            print(f"[MAGMA AGENT] Loading QWEN Commander Model : {commander_id}")
            return QwenCommander(commander_id, cpu_load=optimize_memory, **(qwen_options or {}))
        elif "smollm" in normalized_commander_id:
            print("[MAGMA AGENT] Loading SmolLM3")
            return SmolLMCommander(commander_id, optimize_memory)
        elif "oss" in normalized_commander_id or "gpt_oss" in normalized_commander_id:
            print(f"[MAGMA AGENT] Loading GPT OSS Commander Model : {commander_id}")
            return OSSCommander(commander_id, cpu_load=optimize_memory)
        else:
            print(f"[MAGMA AGENT] Loading Commander Model : {commander_id} with output format {commander_output}")
            return MagmaCommander(cpu_load=optimize_memory, model_id=commander_id, output_style=commander_output, overriding_chat_template_path=chat_template)
            
def load_tsm(tsm_id : str, optimize_memory : bool):
    if tsm_id != "none":
        print(f"[MAGMA AGENT] Loading TSM Model : {tsm_id}")
        return MagmaTSM(cpu_load=optimize_memory, model_id=tsm_id)

    return None
