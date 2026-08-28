from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from magma_core.protocol.agent import AgentOutput, AgentRequest
from magma_core.tsr_engine import (
    DispatcherMode,
    ReactiveTaskState,
    append_dispatcher_tool_calls,
    append_environment_feedback,
    dump_tsm_actions,
    finalize_dispatcher_turn,
    load_tsm_actions_text,
    normalize_dispatcher_history,
    prepare_dispatcher_turn,
    render_dispatcher_history,
    render_dispatcher_view,
    render_tsm_view,
)

from magma_agent.models import (
    BaseModelClient,
    BatchedMessageDispatcher,
    BatchedMessageTSM,
)

from .base import BaseAgent, ModelCall


class TsmInstruction(BaseModel):
    role: Literal["user", "system"]
    content: str


class TaskStateReactiveInput(BaseModel):
    task_state: dict[str, Any]
    call_tsm: bool
    instruction: TsmInstruction | None = None
    environment_feedback: list[str] = Field(default_factory=list)
    persistent_rules: list[str] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)
    dispatcher_history: list[dict[str, Any]] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    inference_mode: bool = False

    @model_validator(mode="after")
    def validate_route(self) -> "TaskStateReactiveInput":
        if self.call_tsm and self.instruction is None:
            raise ValueError("call_tsm=true requires an instruction.")
        if not self.call_tsm and self.instruction is not None:
            raise ValueError("call_tsm=false does not accept a TSM instruction.")
        if self.call_tsm and self.environment_feedback:
            raise ValueError(
                "A TSM instruction and environment feedback cannot share one turn."
            )
        if any(not rule for rule in self.persistent_rules):
            raise ValueError("Persistent rules must be non-empty strings.")
        if any(not message for message in self.environment_feedback):
            raise ValueError("Environment feedback must be non-empty strings.")
        return self


class TaskStateReactiveAgent(BaseAgent):
    """Orchestrate TSR models while delegating state semantics to the core."""

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
        inference_modes = {item.inference_mode for item in parsed_inputs}
        if len(inference_modes) > 1:
            raise ValueError(
                "All inputs in one batch must use the same inference mode."
            )
        inference_mode = inference_modes.pop() if inference_modes else False

        state_candidates: list[dict[str, Any]] = []
        tsm_pending: list[dict[str, Any]] = []
        tsm_messages: dict[str, list[Any]] = {
            "permanent_rules": [],
            "rules": [],
            "goals": [],
            "instruction": [],
        }

        for input_index, (entry, agent_input) in enumerate(
            zip(request.inputs, parsed_inputs)
        ):
            state = ReactiveTaskState.from_dict(agent_input.task_state)
            state_before = state.to_dict()
            history = normalize_dispatcher_history(
                agent_input.dispatcher_history
            )
            if not agent_input.call_tsm:
                state_candidates.append({
                    "input_index": input_index,
                    "source_id": entry.id,
                    "state_index": 0,
                    "state_before": state_before,
                    "state": state,
                    "history": history,
                    "tsm": {
                        "called": False,
                        "view": None,
                        "raw_output": None,
                        "actions": [],
                    },
                })
                continue

            view = state.tsm_view()
            rendered = render_tsm_view(view.representation)
            instruction = agent_input.instruction
            if instruction is None:
                raise RuntimeError("Validated TSM input lost its instruction.")
            for state_index in range(request.candidate_counts["tsm"]):
                tsm_pending.append({
                    "input_index": input_index,
                    "source_id": entry.id,
                    "state_index": state_index,
                    "state_before": state_before,
                    "state": state,
                    "history": history,
                    "view": view,
                    "instruction": instruction,
                })
                tsm_messages["permanent_rules"].append(
                    agent_input.persistent_rules.copy()
                )
                tsm_messages["rules"].append(list(rendered.rules))
                tsm_messages["goals"].append(list(rendered.goals))
                tsm_messages["instruction"].append(
                    instruction.content
                )

        invalid_results: list[tuple[int, int, int, int, dict[str, Any], bool]] = []
        if tsm_pending:
            raw_tsm_outputs = await model_call(
                self.tsm,
                BatchedMessageTSM(**tsm_messages),
                inference_mode,
            )
            if len(raw_tsm_outputs) != len(tsm_pending):
                raise ValueError(
                    f"TSM returned {len(raw_tsm_outputs)} output(s) for "
                    f"{len(tsm_pending)} candidate(s)."
                )
            tsm_validities: list[bool] = []
            for pending, raw_output in zip(tsm_pending, raw_tsm_outputs):
                view = pending["view"]
                try:
                    if not isinstance(raw_output, str):
                        raise TypeError("TSM model output must be raw text.")
                    actions = load_tsm_actions_text(raw_output, view)
                    updated_state = pending["state"].apply_updates(actions)
                    instruction = pending["instruction"]
                    tsm_record = {
                        "called": True,
                        "instruction": {
                            "role": instruction.role,
                            "content": instruction.content,
                        },
                        "view": deepcopy(view.representation),
                        "raw_output": raw_output,
                        "actions": dump_tsm_actions(actions, view),
                    }
                    state_candidates.append({
                        **pending,
                        "state": updated_state,
                        "tsm": tsm_record,
                    })
                    tsm_validities.append(True)
                except (TypeError, ValueError, IndexError) as error:
                    tsm_validities.append(False)
                    output = self._error_output(
                        state_before=pending["state_before"],
                        state_after_tsm=pending["state_before"],
                        history=pending["history"],
                        component="tsm",
                        reason=str(error),
                        raw_output=raw_output,
                        tsm={
                            "called": True,
                            "view": deepcopy(view.representation),
                            "raw_output": raw_output,
                            "actions": None,
                            "error": str(error),
                        },
                    )
                    invalid_results.append((
                        pending["input_index"],
                        pending["state_index"],
                        -1,
                        pending["source_id"],
                        output,
                        False,
                    ))
            self.tsm.update_prompt_log_validity(tsm_validities)

        dispatcher_pending: list[dict[str, Any]] = []
        dispatcher_messages: dict[str, list[Any]] = {
            "mode": [],
            "permanent_rules": [],
            "rules": [],
            "goals": [],
            "attributes": [],
            "history": [],
            "tools": [],
        }
        idle_results: list[tuple[int, int, int, int, dict[str, Any], bool]] = []

        for candidate in state_candidates:
            agent_input = parsed_inputs[candidate["input_index"]]
            history = append_environment_feedback(
                candidate["history"],
                agent_input.environment_feedback,
            )
            preparation = prepare_dispatcher_turn(candidate["state"])
            if preparation.mode is DispatcherMode.IDLE:
                final_state = candidate["state"].to_dict()
                idle_results.append((
                    candidate["input_index"],
                    candidate["state_index"],
                    0,
                    candidate["source_id"],
                    {
                        "task_state": {
                            "before": candidate["state_before"],
                            "after_tsm": final_state,
                            "final": final_state,
                        },
                        "tsm": candidate["tsm"],
                        "dispatcher": {
                            "called": False,
                            "mode": "idle",
                            "view": None,
                            "raw_output": None,
                            "output": None,
                        },
                        "dispatcher_history": history,
                    },
                    True,
                ))
                continue

            if preparation.view is None:
                raise RuntimeError("Non-idle Dispatcher mode lost its view.")
            rendered = render_dispatcher_view(
                preparation.view.representation
            )
            for dispatcher_index in range(
                request.candidate_counts["dispatcher"]
            ):
                dispatcher_pending.append({
                    **candidate,
                    "dispatcher_index": dispatcher_index,
                    "history": history,
                    "preparation": preparation,
                })
                dispatcher_messages["mode"].append(preparation.mode.value)
                dispatcher_messages["permanent_rules"].append(
                    agent_input.persistent_rules.copy()
                )
                dispatcher_messages["rules"].append(list(rendered.rules))
                dispatcher_messages["goals"].append(list(rendered.goals))
                dispatcher_messages["attributes"].append(
                    deepcopy(agent_input.attributes)
                )
                dispatcher_messages["history"].append(
                    render_dispatcher_history(history)
                )
                dispatcher_messages["tools"].append(
                    deepcopy(agent_input.tools)
                )

        raw_dispatcher_outputs: list[Any] = []
        if dispatcher_pending:
            raw_dispatcher_outputs = await model_call(
                self.dispatcher,
                BatchedMessageDispatcher(**dispatcher_messages),
                inference_mode,
            )
            if len(raw_dispatcher_outputs) != len(dispatcher_pending):
                raise ValueError(
                    f"Dispatcher returned {len(raw_dispatcher_outputs)} "
                    f"output(s) for {len(dispatcher_pending)} candidate(s)."
                )

        results = [*invalid_results, *idle_results]
        dispatcher_validities: list[bool] = []
        for pending, raw_output in zip(
            dispatcher_pending,
            raw_dispatcher_outputs,
        ):
            preparation = pending["preparation"]
            try:
                normalized_output = self._normalize_dispatcher_output(
                    raw_output,
                    preparation.mode,
                )
                final_state = finalize_dispatcher_turn(
                    pending["state"],
                    preparation,
                    normalized_output["completed_todos"],
                )
                if preparation.mode is DispatcherMode.EXECUTION_REPORT:
                    final_history: list[dict[str, Any]] = []
                elif "tools" in normalized_output:
                    final_history = append_dispatcher_tool_calls(
                        pending["history"],
                        normalized_output["tools"],
                    )
                else:
                    final_history = deepcopy(pending["history"])
                result_output = {
                    "task_state": {
                        "before": pending["state_before"],
                        "after_tsm": pending["state"].to_dict(),
                        "final": final_state.to_dict(),
                    },
                    "tsm": pending["tsm"],
                    "dispatcher": {
                        "called": True,
                        "mode": preparation.mode.value,
                        "view": deepcopy(preparation.view.representation),
                        "raw_output": deepcopy(raw_output),
                        "output": normalized_output,
                    },
                    "dispatcher_history": final_history,
                }
                valid = True
                dispatcher_validities.append(True)
            except (TypeError, ValueError, IndexError) as error:
                result_output = self._error_output(
                    state_before=pending["state_before"],
                    state_after_tsm=pending["state"].to_dict(),
                    history=pending["history"],
                    component="dispatcher",
                    reason=str(error),
                    raw_output=raw_output,
                    tsm=pending["tsm"],
                    dispatcher={
                        "called": True,
                        "mode": preparation.mode.value,
                        "view": deepcopy(preparation.view.representation),
                        "raw_output": deepcopy(raw_output),
                        "output": None,
                        "error": str(error),
                    },
                )
                valid = False
                dispatcher_validities.append(False)
            results.append((
                pending["input_index"],
                pending["state_index"],
                pending["dispatcher_index"],
                pending["source_id"],
                result_output,
                valid,
            ))
        self.dispatcher.update_prompt_log_validity(dispatcher_validities)

        results.sort(key=lambda item: item[:3])
        candidate_indices: dict[int, int] = {}
        agent_outputs: list[AgentOutput] = []
        for _, _, _, source_id, output, valid in results:
            candidate_index = candidate_indices.get(source_id, 0)
            candidate_indices[source_id] = candidate_index + 1
            agent_outputs.append(AgentOutput(
                source_id=source_id,
                candidate_index=candidate_index,
                valid=valid,
                output=output,
            ))
        return agent_outputs

    @staticmethod
    def _normalize_dispatcher_output(
        raw_output: Any,
        mode: DispatcherMode,
    ) -> dict[str, Any]:
        if not isinstance(raw_output, dict):
            raise ValueError("Dispatcher output must be a parsed JSON object.")
        allowed = {"message", "tools", "completed_todos"}
        if set(raw_output) - allowed:
            raise ValueError("Dispatcher output contains unsupported fields.")
        if "message" in raw_output and "tools" in raw_output:
            raise ValueError("Dispatcher output cannot mix message and tools.")
        if "message" not in raw_output and "tools" not in raw_output:
            raise ValueError("Dispatcher output requires message or tools.")

        completed = raw_output.get("completed_todos", [])
        if not isinstance(completed, list) or not all(
            isinstance(label, str) for label in completed
        ):
            raise TypeError("completed_todos must contain tN labels.")

        output: dict[str, Any] = {"completed_todos": completed.copy()}
        if "message" in raw_output:
            message = raw_output["message"]
            if not isinstance(message, dict) or set(message) != {
                "recipient",
                "content",
            }:
                raise ValueError(
                    "Dispatcher messages require recipient and content."
                )
            if message.get("recipient") not in {"user", "tsm"}:
                raise ValueError("Message recipient must be user or tsm.")
            if not isinstance(message.get("content"), str) or not message["content"]:
                raise ValueError("Message content must be a non-empty string.")
            output["message"] = deepcopy(message)
        else:
            tools = raw_output["tools"]
            if not isinstance(tools, list):
                raise TypeError("Dispatcher tools must be a list.")
            append_dispatcher_tool_calls([], tools)
            output["tools"] = deepcopy(tools)

        if mode is DispatcherMode.EXECUTION_REPORT:
            if "message" not in output or output["message"]["recipient"] != "user":
                raise ValueError("An execution report requires a user message.")
            if output["completed_todos"]:
                raise ValueError("An execution report cannot complete todos.")
        return output

    @staticmethod
    def _error_output(
        *,
        state_before: dict[str, Any],
        state_after_tsm: dict[str, Any],
        history: list[dict[str, Any]],
        component: Literal["tsm", "dispatcher"],
        reason: str,
        raw_output: Any,
        tsm: dict[str, Any],
        dispatcher: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "task_state": {
                "before": deepcopy(state_before),
                "after_tsm": deepcopy(state_after_tsm),
                "final": deepcopy(state_after_tsm),
            },
            "tsm": deepcopy(tsm),
            "dispatcher": dispatcher or {
                "called": False,
                "mode": None,
                "view": None,
                "raw_output": None,
                "output": None,
            },
            "dispatcher_history": deepcopy(history),
            "error": {
                "component": component,
                "reason": reason,
                "raw_output": deepcopy(raw_output),
            },
        }
