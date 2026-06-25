from typing import Any, Dict

from . import MagmaDispatcher, QwenDispatcher


def load_dispatcher(
    name: str,
    model_id: str,
    endpoint: str,
    optimize_memory: bool,
    options: Dict[str, Any],
):
    normalized_model_id = model_id.lower()
    if "qwen" in normalized_model_id:
        supported_options = {
            "quantization_mode",
            "max_new_tokens",
            "attn_implementation",
            "use_cache",
            "enable_thinking",
            "device_map",
            "gpu_memory_limit",
            "allow_cpu_offload",
            "offload_folder",
        }
        unsupported_options = sorted(set(options) - supported_options)
        if unsupported_options:
            print(
                "[MAGMA AGENT] Ignoring unsupported Qwen Dispatcher options: "
                f"{unsupported_options}"
            )

        print(f"[MAGMA AGENT] Loading QWEN Dispatcher Model: {model_id}")
        return QwenDispatcher(
            model_id,
            cpu_load=optimize_memory,
            name=name,
            endpoint=endpoint,
            quantization_mode=options.get("quantization_mode", "4bit"),
            max_new_tokens=options.get("max_new_tokens", 600),
            attn_implementation=options.get("attn_implementation", "sdpa"),
            use_cache=options.get("use_cache", True),
            enable_thinking=options.get("enable_thinking", False),
            device_map=options.get("device_map", "auto"),
            gpu_memory_limit=options.get("gpu_memory_limit"),
            allow_cpu_offload=options.get("allow_cpu_offload", False),
            offload_folder=options.get(
                "offload_folder",
                "/tmp/magma_agent_qwen_dispatcher_offload",
            ),
        )

    supported_options = {"chat_template", "max_new_tokens"}
    unsupported_options = sorted(set(options) - supported_options)
    if unsupported_options:
        print(
            "[MAGMA AGENT] Ignoring unsupported Dispatcher options: "
            f"{unsupported_options}"
        )

    print(f"[MAGMA AGENT] Loading Dispatcher Model: {model_id}")
    return MagmaDispatcher(
        model_id=model_id,
        cpu_load=optimize_memory,
        name=name,
        endpoint=endpoint,
        overriding_chat_template_path=options.get("chat_template"),
        max_new_tokens=options.get("max_new_tokens", 2500),
    )
