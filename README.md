# Audio Noise Removal Service - All-in-One

[![Docker Hub](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/neosun/noise-removal)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI-powered audio noise removal service with Swagger API documentation. Built on ResembleAI's DeepFilterNet model.

## 🚀 Quick Start

```bash
# Pull the all-in-one image
docker pull neosun/noise-removal:v1.0-allinone

# Run the service
docker run -d \
  --name noise-removal \
  --gpus all \
  -p 5080:5080 \
  neosun/noise-removal:v1.0-allinone

# Wait 30 seconds for service to start
# Access Swagger UI: http://0.0.0.0:5080/docs/
```

That's it! No build required, no external dependencies, everything is included.

## 📚 API Documentation

- **Swagger UI**: http://0.0.0.0:5080/docs/
- **API Spec**: http://0.0.0.0:5080/swagger.json
- **Health Check**: http://0.0.0.0:5080/health

## 🎯 Features

- ✅ **All-in-One Image**: Pre-loaded DeepFilterNet model (9.25GB)
- ✅ **Zero Configuration**: Works out of the box
- ✅ **Swagger Documentation**: Interactive API docs
- ✅ **GPU Accelerated**: CUDA 12.4 support
- ✅ **Async Processing**: Support for long audio files
- ✅ **Health Monitoring**: Built-in health check

## 🛠️ Requirements

### Hardware
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **RAM**: 8GB+ recommended

### Software
- **NVIDIA Driver**: 525.60.13+
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: Required

## 📋 API Usage

### Synchronous Processing

```bash
curl -X POST http://0.0.0.0:5080/api \
  -F "audio=@noisy_audio.wav" \
  -o clean_audio.wav
```

### Python Example

```python
import requests

url = "http://0.0.0.0:5080/api"
files = {"audio": open("noisy_audio.wav", "rb")}
response = requests.post(url, files=files)

with open("clean_audio.wav", "wb") as f:
    f.write(response.content)
```

## 🎨 Supported Formats

WAV, MP3, FLAC, OGG, M4A

## 📊 Performance

- **Speed**: ~10x faster than real-time on L40S GPU
- **Memory**: ~2GB VRAM per request
- **Startup**: <5 seconds

## 📄 License

MIT License

---

**Docker Hub**: https://hub.docker.com/r/neosun/noise-removal
