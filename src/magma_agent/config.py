from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    commander_id: Optional[str] = None
    memorizer_id: Optional[str] = None
    commander_output_style: str = "qwen_format"
    commander_chat_template: Optional[str] = None
    optimize_memory: bool = False
    log_file: str = "server.log"
    host: str = "0.0.0.0"
    port: int = 8888
