from typing import Any, Dict

from . import MagmaCommander, OSSCommander, QwenCommander, SmolLMCommander


def load_commander(
    name: str,
    model_id: str,
    endpoint: str,
    optimize_memory: bool,
    options: Dict[str, Any],
):
    normalized_model_id = model_id.lower()
    if "qwen" in normalized_model_id:
        print(f"[MAGMA AGENT] Loading QWEN Commander Model: {model_id}")
        return QwenCommander(
            model_id,
            cpu_load=optimize_memory,
            name=name,
            endpoint=endpoint,
            quantization_mode=options.get("quantization_mode", "4bit"),
            max_new_tokens=options.get("max_new_tokens", 1500),
            attn_implementation=options.get("attn_implementation", "sdpa"),
            use_cache=options.get("use_cache", True),
            enable_thinking=options.get("enable_thinking", False),
            device_map=options.get("device_map", "auto"),
            gpu_memory_limit=options.get("gpu_memory_limit"),
            allow_cpu_offload=options.get("allow_cpu_offload", False),
            offload_folder=options.get("offload_folder", "/tmp/magma_agent_qwen_offload"),
        )

    if "smollm" in normalized_model_id:
        print(f"[MAGMA AGENT] Loading SmolLM Commander Model: {model_id}")
        return SmolLMCommander(
            model_id,
            cpu_load=optimize_memory,
            name=name,
            endpoint=endpoint,
        )

    if "oss" in normalized_model_id or "gpt_oss" in normalized_model_id:
        print(f"[MAGMA AGENT] Loading GPT OSS Commander Model: {model_id}")
        return OSSCommander(
            model_id,
            cpu_load=optimize_memory,
            name=name,
            endpoint=endpoint,
        )

    print(f"[MAGMA AGENT] Loading Commander Model: {model_id}")
    return MagmaCommander(
        model_id=model_id,
        cpu_load=optimize_memory,
        name=name,
        endpoint=endpoint,
        output_style=options.get("output_style", "qwen_format"),
        overriding_chat_template_path=options.get("chat_template"),
    )
