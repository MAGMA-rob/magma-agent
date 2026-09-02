from copy import deepcopy
import json
from typing import Any, Dict, List, Sequence

from pydantic import field_validator

from magma_core.protocol.agent import AgentOutput, AgentRequest

from magma_agent.clients.commander.history import map_chat_role
from magma_agent.clients.commander.magma_model import MagmaCommander
from magma_agent.clients.commander.messages import BatchedMessageCommander
from magma_agent.clients.summarizer.messages import BatchedMessageSummarizer
from magma_agent.models import BaseModelClient

from .base import ModelCall
from .history_reactive import HistoryReactiveAgent, HistoryReactiveInput


class HistorySummaryReactiveInput(HistoryReactiveInput):
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


class HistorySummaryReactiveAgent(HistoryReactiveAgent):
    def __init__(
        self,
        name: str,
        summarizer: BaseModelClient,
        commander: BaseModelClient,
        max_context_characters: int = 5000,
    ) -> None:
        super().__init__(name, commander)
        if (
            not isinstance(max_context_characters, int)
            or max_context_characters <= 0
        ):
            raise ValueError(
                "max_context_characters must be a positive integer"
            )
        if not isinstance(commander, MagmaCommander):
            raise ValueError(
                "History summary reactive requires a MagmaCommander whose "
                "weights provide the trained chat template."
            )
        if commander.tokenizer.chat_template is None:
            raise ValueError(
                "History summary reactive Commander weights must provide "
                "their trained chat template."
            )
        self.summarizer = summarizer
        self.max_context_characters = max_context_characters

    @property
    def models(self) -> list[BaseModelClient]:
        return [self.summarizer, self.commander]

    @staticmethod
    def _commander_message(
        inputs: Sequence[HistoryReactiveInput],
    ) -> BatchedMessageCommander:
        normalized_inputs = []
        for agent_input in inputs:
            normalized_input = agent_input.model_copy(deep=True)
            normalized_history = []
            for previous_message in normalized_input.history:
                message = deepcopy(previous_message)
                role = map_chat_role(message.get("author"))
                content = message.get(
                    "content",
                    message.get("sentence", ""),
                )
                if content is None:
                    content = ""
                if not isinstance(content, str):
                    raise RuntimeError(
                        f"Get {type(content)} with role {role} : {content}"
                    )

                if role == "assistant" and "<tool_call>" not in content:
                    try:
                        output = json.loads(content)
                    except json.JSONDecodeError:
                        output = None
                    if isinstance(output, dict) and (
                        {"say", "action"} & output.keys()
                    ):
                        say = str(output.get("say", "") or "").strip()
                        action = output.get("action", {})
                        if action:
                            content = (
                                say
                                + "<tool_call>"
                                + json.dumps(action, ensure_ascii=False)
                                + "</tool_call>"
                            )
                        else:
                            content = say
                    else:
                        tool_call_start = content.find("{")
                        if tool_call_start >= 0:
                            say = content[:tool_call_start].rstrip()
                            tool_call = content[tool_call_start:].strip()
                            content = (
                                say
                                + "<tool_call>"
                                + tool_call
                                + "</tool_call>"
                            )

                message["content"] = content
                normalized_history.append(message)
            normalized_input.history = normalized_history
            normalized_inputs.append(normalized_input)

        return HistoryReactiveAgent._commander_message(normalized_inputs)

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
        prompt_lengths = self.commander.count_prompt_characters(
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
            if prompt_length <= self.max_context_characters:
                continue
            if not agent_input.history:
                outputs.extend(self._invalid_outputs(
                    entry.id, candidate_count, "context_budget",
                    "Commander prompt exceeds max_context_characters and has no history to summarize.",
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

        sources = []
        commander_inputs = []
        commander_updates = []
        for input_index in valid_indices:
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
            outputs.extend(self._commander_outputs(
                sources,
                raw_answers,
                [
                    {"summary_update": summary_update}
                    for summary_update in commander_updates
                ],
            ))

        source_order = {
            entry.id: index
            for index, entry in enumerate(request.inputs)
        }
        outputs.sort(key=lambda output: (
            source_order[output.source_id],
            output.candidate_index,
        ))
        return outputs
