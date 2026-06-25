import json
import re
from typing import Any, Dict, Union


THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
UNFINISHED_THINK_RE = re.compile(r"<think>.*$", re.DOTALL)


def parse_dispatcher_output(text: str) -> Union[Dict[str, Any], str]:
    cleaned = THINK_RE.sub("", text.strip())
    cleaned = UNFINISHED_THINK_RE.sub("", cleaned).strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            cleaned = "\n".join(lines[1:-1]).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return text.strip()

    if not isinstance(parsed, dict):
        return text.strip()

    return parsed
