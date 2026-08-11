from typing import Any, Dict

from .base import MagmaSummarizer


def load_summarizer(
    name: str,
    model_id: str,
    endpoint: str,
    optimize_memory: bool,
    options: Dict[str, Any],
) -> MagmaSummarizer:
    unsupported_options = sorted(set(options) - {"max_new_tokens"})
    if unsupported_options:
        print(
            f"[MAGMA AGENT] Ignoring unsupported Summarizer options: "
            f"{unsupported_options}"
        )
    max_new_tokens = options.get("max_new_tokens", 1024)
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("Summarizer max_new_tokens must be a positive integer.")
    return MagmaSummarizer(
        model_id=model_id,
        cpu_load=optimize_memory,
        max_new_tokens=max_new_tokens,
        name=name,
        endpoint=endpoint,
    )
