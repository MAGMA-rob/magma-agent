from typing import Any, Dict

from . import MagmaTSM


def load_tsm(
    name: str,
    model_id: str,
    endpoint: str,
    optimize_memory: bool,
    options: Dict[str, Any],
):
    if options:
        print(f"[MAGMA AGENT] Ignoring unsupported TSM options: {sorted(options)}")

    print(f"[MAGMA AGENT] Loading TSM Model: {model_id}")
    return MagmaTSM(
        model_id=model_id,
        cpu_load=optimize_memory,
        name=name,
        endpoint=endpoint,
    )
