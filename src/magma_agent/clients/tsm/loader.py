from typing import Any, Dict

from . import MagmaTSM


def load_tsm(
    name: str,
    model_id: str,
    endpoint: str,
    optimize_memory: bool,
    options: Dict[str, Any],
):
    unsupported_options = sorted(set(options) - {"chat_template"})
    if unsupported_options:
        print(f"[MAGMA AGENT] Ignoring unsupported TSM options: {unsupported_options}")

    print(f"[MAGMA AGENT] Loading TSM Model: {model_id}")
    return MagmaTSM(
        model_id=model_id,
        cpu_load=optimize_memory,
        name=name,
        endpoint=endpoint,
        overriding_chat_template_path=options.get("chat_template"),
    )
