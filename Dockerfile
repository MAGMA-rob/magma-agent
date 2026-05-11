FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH 

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    python3-pip \
    python3.10-dev \
    build-essential \
    libpq-dev \
    gcc \
    curl \
    git \
    bash && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    torch \
    --pre --index-url https://download.pytorch.org/whl/nightly/cu124

WORKDIR /app

COPY pyproject.toml .
RUN pip install --upgrade pip
COPY src ./src


RUN pip install .

CMD [ "python3", "-m", "magma_agent"]