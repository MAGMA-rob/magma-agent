import argparse
import uvicorn

from .config import Settings
from .app import create_app


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--commander-id")
    parser.add_argument("--memorizer-id")
    parser.add_argument("--optimize-memory", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--log-file")

    args = parser.parse_args()

    # Load settings from environment first
    settings = Settings()

    # Override with CLI if provided
    for key, value in vars(args).items():
        if value is not None:
            setattr(settings, key.replace("-", "_"), value)

    app = create_app(settings)

    uvicorn.run(app, host=settings.host, port=settings.port)
