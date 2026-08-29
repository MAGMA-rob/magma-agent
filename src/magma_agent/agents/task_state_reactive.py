from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from magma_core.protocol.agent import AgentOutput, AgentRequest
from magma_core.protocol.tsr import (
    TaskStateReactiveInput,
    TaskStateReactiveResult,
)
from magma_core.tsr_engine import (
    DispatcherMode,
    ReactiveTaskState,
    append_environment_feedback,
    apply_dispatcher_output,
    dump_tsm_actions,
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
                            "instruction": {
                                "role": pending["instruction"].role,
                                "content": pending["instruction"].content,
                            },
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
                            "input_history": history,
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
                application = apply_dispatcher_output(
                    pending["state"],
                    preparation,
                    pending["history"],
                    raw_output,
                )
                result_output = {
                    "task_state": {
                        "before": pending["state_before"],
                        "after_tsm": pending["state"].to_dict(),
                        "final": application.task_state.to_dict(),
                    },
                    "tsm": pending["tsm"],
                    "dispatcher": {
                        "called": True,
                        "mode": preparation.mode.value,
                        "view": deepcopy(preparation.view.representation),
                        "input_history": deepcopy(pending["history"]),
                        "raw_output": deepcopy(raw_output),
                        "output": application.output,
                    },
                    "dispatcher_history": application.history,
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
                        "input_history": deepcopy(pending["history"]),
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
            validated_output = TaskStateReactiveResult.model_validate(output)
            agent_outputs.append(AgentOutput(
                source_id=source_id,
                candidate_index=candidate_index,
                valid=valid,
                output=validated_output.model_dump(mode="python"),
            ))
        return agent_outputs

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
                "input_history": deepcopy(history),
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
