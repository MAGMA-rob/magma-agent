import json
import re
from typing import Any, Dict, Union


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
TOOLS_OUTPUT_RE = re.compile(
    r"\s*(?P<tools>(?:<tool_call>\s*.*?\s*</tool_call>\s*)+)"
    r"<completed>\s*(?P<completed>.*?)\s*</completed>\s*",
    re.DOTALL,
)
MESSAGE_OUTPUT_RE = re.compile(
    r"\s*<message>\s*(?P<message>.*?)\s*</message>\s*"
    r"<completed>\s*(?P<completed>.*?)\s*</completed>\s*",
    re.DOTALL,
)
NOOP_OUTPUT_RE = re.compile(r"\s*<noop>\s*")


def parse_dispatcher_output(text: str) -> Union[Dict[str, Any], str]:
    if NOOP_OUTPUT_RE.fullmatch(text):
        return {"tools": [], "completed_todos": []}

    tools_match = TOOLS_OUTPUT_RE.fullmatch(text)
    message_match = MESSAGE_OUTPUT_RE.fullmatch(text)
    if tools_match is None and message_match is None:
        return text.strip()

    match = tools_match if tools_match is not None else message_match
    if match is None:
        return text.strip()
    try:
        completed_todos = json.loads(match.group("completed"))
    except json.JSONDecodeError:
        return text.strip()
    if not isinstance(completed_todos, list):
        return text.strip()

    if tools_match is not None:
        calls = []
        robots = set()
        for tool_match in TOOL_CALL_RE.finditer(tools_match.group("tools")):
            try:
                tool_call = json.loads(tool_match.group(1))
            except json.JSONDecodeError:
                return text.strip()
            if not isinstance(tool_call, dict) or set(tool_call) != {
                "name",
                "arguments",
            }:
                return text.strip()

            qualified_name = tool_call["name"]
            arguments = tool_call["arguments"]
            if not isinstance(qualified_name, str) or "." not in qualified_name:
                return text.strip()
            robot, name = qualified_name.split(".", 1)
            if not robot or not name or not isinstance(arguments, dict):
                return text.strip()
            if robot in robots:
                return text.strip()
            robots.add(robot)
            calls.append(
                {
                    "robot": robot,
                    "name": name,
                    "arguments": arguments,
                }
            )

        return {"tools": calls, "completed_todos": completed_todos}

    try:
        message = json.loads(match.group("message"))
    except json.JSONDecodeError:
        return text.strip()
    if not isinstance(message, dict) or set(message) != {"recipient", "content"}:
        return text.strip()
    recipient = message["recipient"]
    if recipient not in {"user", "tsm"} or not isinstance(message["content"], str):
        return text.strip()

    return {"message": message, "completed_todos": completed_todos}
