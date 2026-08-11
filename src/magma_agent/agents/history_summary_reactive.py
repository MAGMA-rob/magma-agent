import json
from copy import deepcopy
from typing import Any, Dict, List

from pydantic import BaseModel, Field, field_validator

from magma_core.protocol.agent import AgentOutput, AgentRequest

from magma_agent.clients.commander.base import BaseCommander
from magma_agent.clients.commander.messages import BatchedMessageCommander
from magma_agent.clients.summarizer.messages import BatchedMessageSummarizer
from magma_agent.models import BaseModelClient

from .base import BaseAgent, ModelCall
from .history_reactive import validate_commander_action


class HistorySummaryReactiveInput(BaseModel):
    memory: Dict[str, Any] = Field(default_factory=dict)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    function: List[Dict[str, Any]] = Field(default_factory=list)
    instruction: str
    instruction_role: str = "USER"
    inference_mode: bool = False
    prediction_mode: str = "tool_select"

    @field_validator("memory")
    @classmethod
    def validate_summary(cls, memory: Dict[str, Any]) -> Dict[str, Any]:
        summary = memory.get("summary", "")
        if summary is None:
            memory = memory.copy()
            memory["summary"] = ""
        elif not isinstance(summary, str):
            raise ValueError("memory['summary'] must be a string when provided")
        return memory


class HistorySummaryReactiveAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        summarizer: BaseModelClient,
        commander: BaseModelClient,
        max_context_tokens: int = 5000,
    ) -> None:
        super().__init__(name)
        if not isinstance(max_context_tokens, int) or max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be a positive integer")
        if not isinstance(commander, BaseCommander) or not hasattr(
            commander, "count_prompt_tokens"
        ):
            raise ValueError(
                "History summary reactive requires a MagmaCommander with prompt counting support."
            )
        self.summarizer = summarizer
        self.commander = commander
        self.max_context_tokens = max_context_tokens

    @property
    def models(self) -> list[BaseModelClient]:
        return [self.summarizer, self.commander]

    @staticmethod
    def _commander_message(
        inputs: List[HistorySummaryReactiveInput],
    ) -> BatchedMessageCommander:
        return BatchedMessageCommander(
            memory=[deepcopy(item.memory) for item in inputs],
            attributes=[deepcopy(item.attributes) for item in inputs],
            history=[deepcopy(item.history) for item in inputs],
            function=[deepcopy(item.function) for item in inputs],
            instruction=[item.instruction for item in inputs],
            instruction_role=[item.instruction_role for item in inputs],
            prediction_mode=inputs[0].prediction_mode if inputs else "tool_select",
        )

    @staticmethod
    def _invalid_outputs(
        source_id: int,
        candidate_count: int,
        component: str,
        reason: str,
        summary_update: Dict[str, Any],
    ) -> List[AgentOutput]:
        return [
            AgentOutput(
                source_id=source_id,
                candidate_index=index,
                valid=False,
                output={
                    "component": component,
                    "reason": reason,
                    "raw_output": "",
                    "summary_update": deepcopy(summary_update),
                },
            )
            for index in range(candidate_count)
        ]

    async def process(
        self,
        request: AgentRequest,
        model_call: ModelCall,
    ) -> list[AgentOutput]:
        if set(request.candidate_counts) != {"commander"}:
            raise ValueError(
                "History summary reactive candidate_counts must contain only 'commander'."
            )
        candidate_count = request.candidate_counts["commander"]
        if not request.inputs:
            return []
        parsed_inputs = [
            HistorySummaryReactiveInput.model_validate(entry.input)
            for entry in request.inputs
        ]
        inference_modes = {item.inference_mode for item in parsed_inputs}
        prediction_modes = {item.prediction_mode for item in parsed_inputs}
        if len(inference_modes) > 1 or len(prediction_modes) > 1:
            raise ValueError(
                "All inputs in a batch must use the same inference and prediction modes."
            )
        inference_mode = inference_modes.pop() if inference_modes else False
        effective_inputs = [item.model_copy(deep=True) for item in parsed_inputs]
        prompt_lengths = self.commander.count_prompt_tokens(
            self._commander_message(effective_inputs)
        )
        summary_updates = []
        summarize_indices = []
        outputs = []
        for index, (entry, agent_input, prompt_length) in enumerate(
            zip(request.inputs, effective_inputs, prompt_lengths)
        ):
            previous_summary = agent_input.memory.get("summary", "")
            summary_update = {
                "called": False,
                "input_summary": previous_summary,
                "summary": previous_summary,
            }
            summary_updates.append(summary_update)
            if prompt_length <= self.max_context_tokens:
                continue
            if not agent_input.history:
                outputs.extend(self._invalid_outputs(
                    entry.id, candidate_count, "context_budget",
                    "Commander prompt exceeds max_context_tokens and has no history to summarize.",
                    summary_update,
                ))
                continue
            summarize_indices.append(index)

        if summarize_indices:
            summaries = await model_call(
                self.summarizer,
                BatchedMessageSummarizer(
                    previous_summary=[
                        effective_inputs[index].memory.get("summary", "")
                        for index in summarize_indices
                    ],
                    history=[
                        deepcopy(effective_inputs[index].history)
                        for index in summarize_indices
                    ],
                ),
                inference_mode,
            )
            if len(summaries) != len(summarize_indices):
                raise ValueError(
                    f"Summarizer returned {len(summaries)} output(s) for "
                    f"{len(summarize_indices)} input(s)."
                )
            for input_index, summary in zip(summarize_indices, summaries):
                agent_input = effective_inputs[input_index]
                update = summary_updates[input_index]
                update["called"] = True
                update["input_history"] = deepcopy(agent_input.history)
                if not isinstance(summary, str) or not summary.strip():
                    outputs.extend(self._invalid_outputs(
                        request.inputs[input_index].id, candidate_count,
                        "summarizer", "Summarizer returned an empty or invalid summary.",
                        update,
                    ))
                    continue
                summary = summary.strip()
                update["summary"] = summary
                agent_input.memory["summary"] = summary
                agent_input.history = []

        invalid_source_ids = {output.source_id for output in outputs}
        valid_indices = [
            index for index, entry in enumerate(request.inputs)
            if entry.id not in invalid_source_ids
        ]
        post_summary_lengths = self.commander.count_prompt_tokens(
            self._commander_message([effective_inputs[index] for index in valid_indices])
        ) if valid_indices else []
        accepted_indices = []
        for input_index, prompt_length in zip(valid_indices, post_summary_lengths):
            if prompt_length <= self.max_context_tokens:
                accepted_indices.append(input_index)
                continue
            outputs.extend(self._invalid_outputs(
                request.inputs[input_index].id, candidate_count, "context_budget",
                "Commander prompt still exceeds max_context_tokens after summarization.",
                summary_updates[input_index],
            ))

        sources = []
        commander_inputs = []
        commander_updates = []
        for input_index in accepted_indices:
            for _ in range(candidate_count):
                sources.append(request.inputs[input_index].id)
                commander_inputs.append(effective_inputs[input_index])
                commander_updates.append(summary_updates[input_index])

        if sources:
            raw_answers = await model_call(
                self.commander,
                self._commander_message(commander_inputs),
                inference_mode,
            )
            if len(raw_answers) != len(sources):
                raise ValueError(
                    f"Commander returned {len(raw_answers)} output(s) for "
                    f"{len(sources)} candidate(s)."
                )
            candidate_indices: Dict[int, int] = {}
            validities = []
            for source_id, answer, summary_update in zip(
                sources, raw_answers, commander_updates
            ):
                candidate_index = candidate_indices.get(source_id, 0)
                candidate_indices[source_id] = candidate_index + 1
                try:
                    if isinstance(answer, str):
                        answer = json.loads(answer)
                    if not isinstance(answer, dict):
                        raise ValueError("Commander output must be a JSON object")
                    output = {
                        "say": answer.get("say", ""),
                        "action": validate_commander_action(answer.get("action", {})),
                        "summary_update": deepcopy(summary_update),
                    }
                    if "think" in answer:
                        output["think"] = answer["think"]
                    valid = True
                except (TypeError, ValueError, json.JSONDecodeError) as error:
                    output = {
                        "component": "commander",
                        "reason": str(error),
                        "raw_output": answer,
                        "summary_update": deepcopy(summary_update),
                    }
                    valid = False
                validities.append(valid)
                outputs.append(AgentOutput(
                    source_id=source_id,
                    candidate_index=candidate_index,
                    valid=valid,
                    output=output,
                ))
            self.commander.update_prompt_log_validity(validities)

        source_order = {
            entry.id: index
            for index, entry in enumerate(request.inputs)
        }
        outputs.sort(key=lambda output: (
            source_order[output.source_id],
            output.candidate_index,
        ))
        return outputs
