import json
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from magma_core.protocol.agent import AgentOutput, AgentRequest

from magma_agent.models import (
    BaseModelClient,
    BatchedMessageDispatcher,
    BatchedMessageTSM,
)

from .base import BaseAgent, ModelCall


REPRESENTATION_FIELDS = ("rules", "goals", "todo")


class TaskStateInstruction(BaseModel):
    type: Literal["message", "tool_result"]
    content: str


class TaskStateReactiveInput(BaseModel):
    memory: Dict[str, Any]
    attributes: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    function: List[Dict[str, Any]] = Field(default_factory=list)
    instruction: TaskStateInstruction
    completed_todos: List[str] = Field(default_factory=list)
    completed_goals: List[str] = Field(default_factory=list)
    inference_mode: bool = False
    tsm: Optional[Dict[str, Any]] = None


def normalize_representation(memory: Dict[str, Any]) -> Dict[str, Any]:
    representation = deepcopy(memory)
    for field_name in REPRESENTATION_FIELDS:
        value = representation.get(field_name, [])
        if value is None:
            value = []
        if not isinstance(value, list):
            raise TypeError(f"memory[{field_name!r}] must be a list")
        if any(not isinstance(item, str) for item in value):
            raise TypeError(f"memory[{field_name!r}] must contain only strings")
        representation[field_name] = value.copy()
    return representation


def apply_task_state_update(
    memory: Dict[str, Any],
    update: Any,
) -> Dict[str, Any]:
    representation = normalize_representation(memory)
    if isinstance(update, str):
        raise ValueError(f"Cannot apply malformed TSM update: {update}")
    if not isinstance(update, list):
        raise TypeError(f"TSM update must be a list, got {type(update).__name__}")

    action_targets = {
        "add_goal": ("goals", "add"),
        "remove_goal": ("goals", "remove"),
        "add_rule": ("rules", "add"),
        "remove_rule": ("rules", "remove"),
        "add_todo": ("todo", "add"),
        "remove_todo": ("todo", "remove"),
    }
    removals: Dict[str, List[int]] = {
        field_name: []
        for field_name in REPRESENTATION_FIELDS
    }
    additions: List[tuple[str, str]] = []

    for action in update:
        if not isinstance(action, dict):
            raise TypeError(f"TSM action must be a dict, got {type(action).__name__}")
        if "type" not in action or "content" not in action:
            raise ValueError("TSM action must contain 'type' and 'content'")
        action_type = action["type"]
        if action_type not in action_targets:
            raise ValueError(f"Unknown TSM action type {action_type!r}")

        field_name, operation = action_targets[action_type]
        content = action["content"]
        if operation == "add":
            if not isinstance(content, str):
                raise TypeError(f"{action_type} content must be a string")
            additions.append((field_name, content))
            continue

        if not isinstance(content, int):
            raise TypeError(f"{action_type} content must be an integer index")
        if content < 0 or content >= len(representation[field_name]):
            raise IndexError(
                f"{action_type} index {content} is out of range for {field_name}"
            )
        if content in removals[field_name]:
            raise ValueError(
                f"{action_type} index {content} is removed more than once"
            )
        removals[field_name].append(content)

    for field_name, indices in removals.items():
        for index in sorted(indices, reverse=True):
            representation[field_name].pop(index)
    for field_name, content in additions:
        representation[field_name].append(content)

    return normalize_representation(representation)


def dispatcher_representation(
    input_representation: Dict[str, Any],
    representation: Dict[str, Any],
) -> Dict[str, Any]:
    result = deepcopy(representation)
    previous_rules = input_representation.get("rules", [])
    current_rules = representation["rules"]
    new_rules = {
        rule for rule in current_rules
        if rule not in previous_rules
    }
    result["rules"] = [
        f"[removed] {rule}"
        for rule in previous_rules
        if rule not in current_rules
    ]
    result["rules"].extend(
        f"[new] {rule}" if rule in new_rules else rule
        for rule in current_rules
    )

    completed_todos = set(representation.get("completed_todos", []))
    result["todo"] = [
        f"[done] {todo}" if todo in completed_todos else todo
        for todo in representation["todo"]
    ]
    return result


def parse_dispatcher_output(
    raw_output: Any,
    todo_count: Optional[int] = None,
) -> Dict[str, Any]:
    if isinstance(raw_output, str):
        raw_output = json.loads(raw_output)
    if not isinstance(raw_output, dict):
        raise ValueError("Dispatcher output must be a JSON object")

    has_message = "message" in raw_output
    has_tools = "tools" in raw_output
    if has_message and has_tools:
        raise ValueError("Answer cannot contain both message and tools")

    say = ""
    message_from_dispatcher = ""
    calls = []
    if has_message:
        message = raw_output["message"]
        if not isinstance(message, dict):
            raise ValueError("message must be an object")
        recipient = message.get("recipient")
        if recipient not in {"user", "system"}:
            raise ValueError("message recipient must be 'user' or 'system'")
        content = message.get("content", "")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        if recipient == "user":
            say = content
        else:
            message_from_dispatcher = content
    elif has_tools:
        if not isinstance(raw_output["tools"], list):
            raise ValueError("tools must be a list")
        robots = set()
        for call in raw_output["tools"]:
            if not isinstance(call, dict):
                raise ValueError("Tool call must be an object")
            robot = call.get("robot")
            name = call.get("name")
            arguments = call.get("arguments", {})
            if not isinstance(robot, str) or not robot:
                raise ValueError("Tool call must define a non-empty robot")
            if robot in robots:
                raise ValueError(f"Multiple calls for the same robot ({robot})")
            if not isinstance(name, str) or not name:
                raise ValueError(f"Call for target {robot!r} must define a name")
            if not isinstance(arguments, dict):
                raise ValueError(f"Call arguments for target {robot!r} must be an object")
            robots.add(robot)
            calls.append(
                {"robot": robot, "name": name, "arguments": arguments}
            )

    completed_todo = raw_output.get("completed_todos", [])
    if not isinstance(completed_todo, list):
        raise ValueError("completed_todos must be a list")
    completed_todo_ids = []
    for raw_todo_id in completed_todo:
        try:
            todo_id = int(raw_todo_id)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"completed_todos contains a non-integer id: {raw_todo_id!r}"
            ) from error
        if todo_count is not None and not 0 <= todo_id < todo_count:
            raise ValueError(
                f"completed todo id {todo_id} is outside the todo list"
            )
        completed_todo_ids.append(str(todo_id))
    return {
        "say": say,
        "calls": calls,
        "message_from_dispatcher": message_from_dispatcher,
        "completed_todo": completed_todo_ids,
        "raw_output": raw_output,
    }


class TaskStateReactiveAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        tsm: BaseModelClient,
        dispatcher: BaseModelClient,
    ) -> None:
        super().__init__(name)
        self.tsm = tsm
        self.dispatcher = dispatcher

    @property
    def models(self) -> list[BaseModelClient]:
        return [self.tsm, self.dispatcher]

    async def process(
        self,
        request: AgentRequest,
        model_call: ModelCall,
    ) -> list[AgentOutput]:
        if set(request.candidate_counts) != {"tsm", "dispatcher"}:
            raise ValueError(
                "Task state reactive candidate_counts must contain 'tsm' and "
                "'dispatcher'."
            )

        parsed_inputs = [
            TaskStateReactiveInput.model_validate(entry.input)
            for entry in request.inputs
        ]
        modes = {item.inference_mode for item in parsed_inputs}
        if len(modes) > 1:
            raise ValueError(
                "All inputs in a batch must use the same inference mode."
            )
        inference_mode = modes.pop() if modes else False

        state_candidates: list[Dict[str, Any]] = []
        tsm_sources = []
        tsm_permanent_rules = []
        tsm_goals = []
        tsm_rules = []
        tsm_todo = []
        tsm_instructions = []

        for input_index, (entry, agent_input) in enumerate(
            zip(request.inputs, parsed_inputs)
        ):
            input_representation = normalize_representation(agent_input.memory)
            completed_todos = list(
                input_representation.get("completed_todos", [])
            )
            for raw_todo_id in agent_input.completed_todos:
                try:
                    todo_id = int(raw_todo_id)
                except (TypeError, ValueError):
                    continue
                if 0 <= todo_id < len(input_representation["todo"]):
                    todo = input_representation["todo"][todo_id]
                    if todo.startswith("[done] "):
                        todo = todo.removeprefix("[done] ")
                        input_representation["todo"][todo_id] = todo
                    if todo not in completed_todos:
                        completed_todos.append(todo)
            input_representation["completed_todos"] = completed_todos

            if agent_input.tsm is not None:
                supplied_state = agent_input.tsm
                state_candidates.append(
                    {
                        "input_index": input_index,
                        "source_id": entry.id,
                        "state_index": 0,
                        "input_representation": normalize_representation(
                            supplied_state["input_representation"]
                        ),
                        "representation": normalize_representation(
                            supplied_state["representation"]
                        ),
                        "called": bool(supplied_state.get("called", False)),
                        "raw_output": supplied_state.get("raw_output"),
                    }
                )
                continue

            if agent_input.instruction.type == "tool_result":
                representation = deepcopy(input_representation)
                state_candidates.append(
                    {
                        "input_index": input_index,
                        "source_id": entry.id,
                        "state_index": 0,
                        "input_representation": input_representation,
                        "representation": representation,
                        "called": False,
                        "raw_output": None,
                    }
                )
                continue

            for state_index in range(request.candidate_counts["tsm"]):
                tsm_sources.append(
                    (
                        input_index,
                        entry.id,
                        state_index,
                        deepcopy(input_representation),
                    )
                )
                permanent_rules = input_representation.get("memory_list", [])
                if permanent_rules is None:
                    permanent_rules = []
                if not isinstance(permanent_rules, list):
                    raise TypeError("memory['memory_list'] must be a list")
                if any(not isinstance(rule, str) for rule in permanent_rules):
                    raise TypeError(
                        "memory['memory_list'] must contain only strings"
                    )
                tsm_permanent_rules.append(permanent_rules.copy())
                goals = [
                    f"[completed] {goal}"
                    for goal in agent_input.completed_goals
                ]
                goals.extend(
                    f"[current] {goal}" if index == 0 else goal
                    for index, goal in enumerate(
                        input_representation["goals"]
                    )
                )
                tsm_goals.append(goals)
                tsm_rules.append(input_representation["rules"].copy())
                tsm_todo.append([
                    f"[done] {todo}"
                    if todo in completed_todos
                    else todo
                    for todo in input_representation["todo"]
                ])
                tsm_instructions.append(agent_input.instruction.content)

        if tsm_sources:
            raw_updates = await model_call(
                self.tsm,
                BatchedMessageTSM(
                    permanent_rules=tsm_permanent_rules,
                    goals=tsm_goals,
                    rules=tsm_rules,
                    todo=tsm_todo,
                    instruction=tsm_instructions,
                ),
                inference_mode,
            )
            if len(raw_updates) != len(tsm_sources):
                raise ValueError(
                    f"TSM returned {len(raw_updates)} output(s) for "
                    f"{len(tsm_sources)} candidate(s)."
                )
            tsm_validities = []
            for source, raw_update in zip(tsm_sources, raw_updates):
                (
                    input_index,
                    source_id,
                    state_index,
                    input_representation,
                ) = source
                try:
                    representation = apply_task_state_update(
                        input_representation,
                        raw_update,
                    )
                    representation["completed_todos"] = [
                        todo
                        for todo in input_representation.get(
                            "completed_todos",
                            [],
                        )
                        if todo in representation["todo"]
                    ]
                    error = None
                    tsm_validities.append(True)
                except (TypeError, ValueError, IndexError) as exception:
                    representation = input_representation
                    error = str(exception)
                    tsm_validities.append(False)
                state_candidates.append(
                    {
                        "input_index": input_index,
                        "source_id": source_id,
                        "state_index": state_index,
                        "input_representation": input_representation,
                        "representation": representation,
                        "called": True,
                        "raw_output": raw_update,
                        "error": error,
                    }
                )
            self.tsm.update_prompt_log_validity(tsm_validities)

        pending_dispatcher = []
        invalid_results = []
        dispatcher_memory = []
        dispatcher_attributes = []
        dispatcher_history = []
        dispatcher_functions = []

        for state in state_candidates:
            agent_input = parsed_inputs[state["input_index"]]
            completed_goals = list(agent_input.completed_goals)
            for goal in state["input_representation"]["goals"]:
                if (
                    goal not in state["representation"]["goals"]
                    and goal not in completed_goals
                ):
                    completed_goals.append(goal)
            state["completed_goals"] = completed_goals

            if state.get("error") is not None:
                invalid_results.append(
                    (
                        state["input_index"],
                        state["state_index"],
                        -1,
                        state["source_id"],
                        {
                            "tsm": {
                                "input_representation": state["input_representation"],
                                "representation": state["representation"],
                                "called": state["called"],
                                "raw_output": state["raw_output"],
                                "format_error": state["error"],
                            },
                            "error": {
                                "component": "tsm",
                                "reason": state["error"],
                                "raw_output": state["raw_output"],
                            },
                            "completed_goals": completed_goals,
                        },
                    )
                )
                continue

            for dispatcher_index in range(
                request.candidate_counts["dispatcher"]
            ):
                pending_dispatcher.append((state, dispatcher_index))
                dispatcher_memory.append(
                    dispatcher_representation(
                        state["input_representation"],
                        state["representation"],
                    )
                )
                dispatcher_attributes.append(agent_input.attributes.copy())
                dispatcher_functions.append(agent_input.function.copy())
                history = (
                    []
                    if state["called"]
                    else deepcopy(agent_input.history)
                )
                if agent_input.instruction.type == "tool_result":
                    feedback_content = agent_input.instruction.content
                    try:
                        feedback = json.loads(feedback_content)
                    except json.JSONDecodeError:
                        feedback = feedback_content

                    if isinstance(feedback, dict):
                        feedback.pop("previous_tool_call", None)
                        feedback_content = json.dumps(
                            feedback,
                            ensure_ascii=False,
                        )

                    history.append({
                        "author": "SYSTEM",
                        "content": feedback_content,
                        "timestamp": 0,
                    })
                dispatcher_history.append(history)

        raw_dispatcher_outputs = []
        if pending_dispatcher:
            raw_dispatcher_outputs = await model_call(
                self.dispatcher,
                BatchedMessageDispatcher(
                    memory=dispatcher_memory,
                    attributes=dispatcher_attributes,
                    history=dispatcher_history,
                    function=dispatcher_functions,
                ),
                inference_mode,
            )
            if len(raw_dispatcher_outputs) != len(pending_dispatcher):
                raise ValueError(
                    f"Dispatcher returned {len(raw_dispatcher_outputs)} output(s) "
                    f"for {len(pending_dispatcher)} candidate(s)."
                )

        results = invalid_results
        dispatcher_validities = []
        for pending, raw_output in zip(
            pending_dispatcher,
            raw_dispatcher_outputs,
        ):
            state, dispatcher_index = pending
            task_state = {
                "input_representation": state["input_representation"],
                "representation": state["representation"],
                "called": state["called"],
                "raw_output": state["raw_output"],
            }
            try:
                dispatcher = parse_dispatcher_output(
                    raw_output,
                    len(task_state["representation"]["todo"]),
                )
                output = {
                    "tsm": task_state,
                    "dispatcher": dispatcher["raw_output"],
                    "completed_goals": state["completed_goals"],
                }
                valid = True
                dispatcher_validities.append(True)
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                valid = False
                dispatcher_validities.append(False)
                output = {
                    "tsm": task_state,
                    "dispatcher": raw_output,
                    "error": {
                        "component": "dispatcher",
                        "reason": str(error),
                        "raw_output": raw_output,
                    },
                    "completed_goals": state["completed_goals"],
                }
            results.append(
                (
                    state["input_index"],
                    state["state_index"],
                    dispatcher_index,
                    state["source_id"],
                    output,
                    valid,
                )
            )
        self.dispatcher.update_prompt_log_validity(
            dispatcher_validities
        )

        normalized_results = []
        for result in results:
            if len(result) == 5:
                input_index, state_index, dispatcher_index, source_id, output = result
                valid = False
            else:
                (
                    input_index,
                    state_index,
                    dispatcher_index,
                    source_id,
                    output,
                    valid,
                ) = result
            normalized_results.append(
                (
                    input_index,
                    state_index,
                    dispatcher_index,
                    source_id,
                    output,
                    valid,
                )
            )

        normalized_results.sort(key=lambda item: item[:3])
        candidate_indices: Dict[int, int] = {}
        outputs = []
        for _, _, _, source_id, output, valid in normalized_results:
            candidate_index = candidate_indices.get(source_id, 0)
            candidate_indices[source_id] = candidate_index + 1
            outputs.append(
                AgentOutput(
                    source_id=source_id,
                    candidate_index=candidate_index,
                    valid=valid,
                    output=output,
                )
            )
        return outputs
