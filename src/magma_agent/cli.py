import argparse

import uvicorn

from .app import create_app
from .config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--config-json")
    parser.add_argument("--optimize-memory", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-file")
    parser.add_argument("--prompt-log-dir")

    args = parser.parse_args()
    if args.config is not None and args.config_json is not None:
        parser.error("--config and --config-json are mutually exclusive")

    overrides = {}
    if args.config is not None:
        overrides["config_path"] = args.config
    if args.config_json is not None:
        overrides["config_json"] = args.config_json
    if args.optimize_memory:
        overrides["optimize_memory"] = True
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.log_file is not None:
        overrides["log_file"] = args.log_file
    if args.prompt_log_dir is not None:
        overrides["prompt_log_dir"] = args.prompt_log_dir

    settings = Settings(**overrides)
    app = create_app(settings)

    uvicorn.run(app, host=settings.host, port=settings.port)
