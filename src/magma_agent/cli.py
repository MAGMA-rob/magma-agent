import argparse

import uvicorn

from .app import create_app
from .config import Settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-config")
    parser.add_argument("--models-json")
    parser.add_argument("--optimize-memory", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-file")

    args = parser.parse_args()
    if args.models_config is not None and args.models_json is not None:
        parser.error("--models-config and --models-json are mutually exclusive")

    overrides = {}
    if args.models_config is not None:
        overrides["magma_models_config"] = args.models_config
    if args.models_json is not None:
        overrides["magma_models_json"] = args.models_json
    if args.optimize_memory:
        overrides["optimize_memory"] = True
    if args.host is not None:
        overrides["host"] = args.host
    if args.port is not None:
        overrides["port"] = args.port
    if args.log_file is not None:
        overrides["log_file"] = args.log_file

    settings = Settings(**overrides)
    app = create_app(settings)

    uvicorn.run(app, host=settings.host, port=settings.port)
