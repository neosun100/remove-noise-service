FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MODELSCOPE_CACHE=/app/models \
    HF_HOME=/app/models \
    HF_HUB_DISABLE_SYMLINKS_WARNING=true

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt --no-deps && \
    pip3 install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

COPY . .

RUN mkdir -p /app/tmp /app/models && chmod -R 777 /app/tmp

EXPOSE 5080

CMD ["python3", "api_enhanced.py"]
