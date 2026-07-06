from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from magma_core.protocol.agent import AgentOutput, AgentRequest

from magma_agent.models import BaseModelClient


ModelCall = Callable[[BaseModelClient, Any, bool], Awaitable[list[Any]]]


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @property
    @abstractmethod
    def models(self) -> list[BaseModelClient]:
        raise NotImplementedError

    @abstractmethod
    async def process(
        self,
        request: AgentRequest,
        model_call: ModelCall,
    ) -> list[AgentOutput]:
        raise NotImplementedError
