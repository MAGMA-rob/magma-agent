import json
from copy import deepcopy
from typing import Any, Dict, List, Sequence

from pydantic import BaseModel, Field

from magma_core.protocol.agent import AgentOutput, AgentRequest

from magma_agent.models import BaseModelClient, BatchedMessageCommander

from .base import BaseAgent, ModelCall


class HistoryReactiveInput(BaseModel):
    memory: Dict[str, Any] = Field(default_factory=dict)
    attributes: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    function: List[Dict[str, Any]] = Field(default_factory=list)
    instruction: str
    instruction_role: str = "USER"
    inference_mode: bool = False
    prediction_mode: str = "tool_select"


def validate_commander_action(action: Any) -> Dict[str, Any]:
    if isinstance(action, str):
        action = json.loads(action)
    if not isinstance(action, dict):
        raise ValueError("Commander action must be a JSON object")

    for robot_name, call_data in action.items():
        if not isinstance(call_data, dict):
            raise ValueError(f"Call for target {robot_name!r} must be an object")
        name = call_data.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"Call for target {robot_name!r} must define a non-empty name"
            )
        if not isinstance(call_data.get("arguments", {}), dict):
            raise ValueError(
                f"Call arguments for target {robot_name!r} must be an object"
            )
    return action


class HistoryReactiveAgent(BaseAgent):
    def __init__(self, name: str, commander: BaseModelClient) -> None:
        super().__init__(name)
        self.commander = commander

    @property
    def models(self) -> list[BaseModelClient]:
        return [self.commander]

    @staticmethod
    def _commander_message(
        inputs: Sequence[HistoryReactiveInput],
    ) -> BatchedMessageCommander:
        return BatchedMessageCommander(
            memory=[deepcopy(item.memory) for item in inputs],
            attributes=[deepcopy(item.attributes) for item in inputs],
            history=[deepcopy(item.history) for item in inputs],
            function=[deepcopy(item.function) for item in inputs],
            instruction=[item.instruction for item in inputs],
            instruction_role=[item.instruction_role for item in inputs],
            prediction_mode=(
                inputs[0].prediction_mode
                if inputs
                else "tool_select"
            ),
        )

    def _commander_outputs(
        self,
        sources: Sequence[int],
        answers: Sequence[Any],
        metadata: Sequence[Dict[str, Any]] | None = None,
    ) -> List[AgentOutput]:
        if len(answers) != len(sources):
            raise ValueError(
                f"Commander returned {len(answers)} output(s) for "
                f"{len(sources)} candidate(s)."
            )
        if metadata is not None and len(metadata) != len(sources):
            raise ValueError(
                "Commander output metadata must match the number of candidates."
            )

        candidate_indices: Dict[int, int] = {}
        outputs = []
        for index, (source_id, answer) in enumerate(zip(sources, answers)):
            candidate_index = candidate_indices.get(source_id, 0)
            candidate_indices[source_id] = candidate_index + 1
            extra_output = deepcopy(metadata[index]) if metadata is not None else {}
            try:
                if isinstance(answer, str):
                    answer = json.loads(answer)
                if not isinstance(answer, dict):
                    raise ValueError("Commander output must be a JSON object")
                output = {
                    "say": answer.get("say", ""),
                    "action": validate_commander_action(answer.get("action", {})),
                    **extra_output,
                }
                if "think" in answer:
                    output["think"] = answer["think"]
                valid = True
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                output = {
                    "component": "commander",
                    "reason": str(error),
                    "raw_output": answer,
                    **extra_output,
                }
                valid = False

            outputs.append(
                AgentOutput(
                    source_id=source_id,
                    candidate_index=candidate_index,
                    valid=valid,
                    output=output,
                )
            )

        self.commander.update_prompt_log_validity(
            [output.valid for output in outputs]
        )
        return outputs

    async def process(
        self,
        request: AgentRequest,
        model_call: ModelCall,
    ) -> list[AgentOutput]:
        if set(request.candidate_counts) != {"commander"}:
            raise ValueError(
                "History reactive candidate_counts must contain only 'commander'."
            )

        candidate_count = request.candidate_counts["commander"]
        parsed_inputs = [
            HistoryReactiveInput.model_validate(entry.input)
            for entry in request.inputs
        ]

        sources = []
        commander_inputs = []

        for entry, agent_input in zip(request.inputs, parsed_inputs):
            for _ in range(candidate_count):
                sources.append(entry.id)
                commander_inputs.append(agent_input)

        if not sources:
            return []

        inference_modes = {item.inference_mode for item in parsed_inputs}
        prediction_modes = {item.prediction_mode for item in parsed_inputs}
        if len(inference_modes) != 1 or len(prediction_modes) != 1:
            raise ValueError(
                "All inputs in a batch must use the same inference and prediction modes."
            )

        answers = await model_call(
            self.commander,
            self._commander_message(commander_inputs),
            inference_modes.pop(),
        )
        return self._commander_outputs(sources, answers)
