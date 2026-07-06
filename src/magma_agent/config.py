import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class ModelSettings(BaseModel):
    name: str
    type: str
    model_id: str
    endpoint: Optional[str] = None
    options: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("endpoint")
    @classmethod
    def validate_endpoint(cls, endpoint: Optional[str]) -> Optional[str]:
        if endpoint is not None and not endpoint.startswith("/"):
            raise ValueError("Model endpoint must start with '/'.")
        return endpoint


class AgentSettings(BaseModel):
    name: str
    type: str
    models: Dict[str, str]


class Settings(BaseSettings):
    models: List[ModelSettings] = Field(default_factory=list)
    agents: List[AgentSettings] = Field(default_factory=list)
    optimize_memory: bool = False
    log_file: str = "server.log"
    host: str = "0.0.0.0"
    port: int = 8888
    prompt_log_dir: Optional[str] = None
    config_path: Optional[str] = None
    config_json: Optional[str] = None

    @model_validator(mode="after")
    def load_models_from_json_sources(self) -> "Settings":
        if self.models and self.agents:
            return self

        raw_config = None
        if self.config_json:
            raw_config = self.config_json
        elif self.config_path:
            raw_config = Path(self.config_path).read_text(encoding="utf-8")

        if raw_config is None:
            return self

        parsed = json.loads(raw_config)
        if not isinstance(parsed, dict):
            raise ValueError("Agent configuration must contain 'models' and 'agents'.")

        self.models = [
            ModelSettings.model_validate(item)
            for item in parsed.get("models", [])
        ]
        self.agents = [
            AgentSettings.model_validate(item)
            for item in parsed.get("agents", [])
        ]
        if self.prompt_log_dir is None:
            self.prompt_log_dir = parsed.get("prompt_log_dir")
        return self
