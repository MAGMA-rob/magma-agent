from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    commander_id: Optional[str] = None
    memorizer_id: Optional[str] = None
    commander_output_style: str = "qwen_format"
    commander_chat_template: Optional[str] = None
    optimize_memory: bool = False
    qwen_quantization: str = "4bit"
    qwen_max_new_tokens: int = 1500
    qwen_attn_implementation: Optional[str] = "sdpa"
    qwen_use_cache: bool = True
    qwen_device_map: str = "auto"
    qwen_gpu_memory_limit: Optional[str] = None
    qwen_allow_cpu_offload: bool = False
    qwen_offload_folder: str = "/tmp/magma_agent_qwen_offload"
    log_file: str = "server.log"
    host: str = "0.0.0.0"
    port: int = 8888
