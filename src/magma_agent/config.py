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


class Settings(BaseSettings):
    models: List[ModelSettings] = Field(default_factory=list)
    optimize_memory: bool = False
    log_file: str = "server.log"
    host: str = "0.0.0.0"
    port: int = 8888
    magma_models_config: Optional[str] = None
    magma_models_json: Optional[str] = None

    @model_validator(mode="after")
    def load_models_from_json_sources(self) -> "Settings":
        if self.models:
            return self

        raw_config = None
        if self.magma_models_json:
            raw_config = self.magma_models_json
        elif self.magma_models_config:
            raw_config = Path(self.magma_models_config).read_text(encoding="utf-8")

        if raw_config is None:
            return self

        parsed = json.loads(raw_config)
        if isinstance(parsed, dict):
            parsed_models = parsed.get("models", [])
        else:
            parsed_models = parsed

        self.models = [ModelSettings.model_validate(item) for item in parsed_models]
        return self
