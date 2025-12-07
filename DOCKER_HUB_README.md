# Audio Noise Removal Service - All-in-One Docker Image

[![Docker Hub](https://img.shields.io/badge/Docker-Hub-blue?logo=docker)](https://hub.docker.com/r/neosun/noise-removal)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

AI-powered audio noise removal service with Swagger API documentation. Built on ResembleAI's DeepFilterNet model.

## 🚀 Quick Start

```bash
docker pull neosun/noise-removal:v1.0-allinone

docker run -d \
  --name noise-removal \
  --gpus all \
  -p 5080:5080 \
  neosun/noise-removal:v1.0-allinone
```

## 📚 API Documentation

- **Swagger UI**: http://localhost:5080/docs/
- **API Spec**: http://localhost:5080/swagger.json
- **Health Check**: http://localhost:5080/health

## 🎯 Features

- ✅ **All-in-One Image**: Pre-loaded DeepFilterNet model (9.25GB)
- ✅ **Swagger Documentation**: Interactive API docs at `/docs/`
- ✅ **GPU Accelerated**: CUDA 12.4 support
- ✅ **Async Processing**: Support for long audio files
- ✅ **Health Monitoring**: Built-in health check endpoint

## 📋 API Endpoints

### 1. Synchronous Processing
```bash
POST /api
Content-Type: multipart/form-data

# Upload audio file
curl -X POST http://localhost:5080/api \
  -F "audio=@input.wav" \
  -o output.wav
```

### 2. Asynchronous Processing
```bash
# Submit task
POST /upload_async
Response: {"task_id": "xxx", "status": "processing"}

# Check status
GET /status/{task_id}
Response: {"status": "completed", "result_url": "..."}
```

### 3. Health Check
```bash
GET /health
Response: {
  "status": "healthy",
  "model_loaded": true,
  "gpu_idle_time": 10,
  "active_tasks": 0
}
```

## 🛠️ Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Driver**: NVIDIA Driver 525.60.13+
- **Docker**: 20.10+
- **NVIDIA Container Toolkit**: Required

## 🔧 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUSTOM_DOMAIN` | `noise.aws.xin` | Custom domain for service |
| `USE_HTTPS` | `true` | Enable HTTPS in URLs |
| `MODELSCOPE_CACHE` | `/app/models` | Model cache directory |

## 📊 Image Details

- **Base Image**: nvidia/cuda:12.4.0-runtime-ubuntu22.04
- **Size**: 9.25GB (includes pre-loaded model)
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu124
- **Model**: DeepFilterNet (pre-downloaded)

## 🎨 Supported Audio Formats

- WAV
- MP3
- FLAC
- OGG
- M4A

## 📝 Example Usage

### Python
```python
import requests

url = "http://localhost:5080/api"
files = {"audio": open("noisy_audio.wav", "rb")}

response = requests.post(url, files=files)

with open("clean_audio.wav", "wb") as f:
    f.write(response.content)
```

### cURL
```bash
curl -X POST http://localhost:5080/api \
  -F "audio=@noisy_audio.wav" \
  -o clean_audio.wav
```

## 🔍 Health Check

```bash
# Docker health check (built-in)
docker inspect --format='{{.State.Health.Status}}' noise-removal

# Manual check
curl http://localhost:5080/health
```

## 📦 Volume Mapping

```bash
docker run -d \
  --name noise-removal \
  --gpus all \
  -p 5080:5080 \
  -v /path/to/models:/app/models \
  -v /path/to/tmp:/app/tmp \
  neosun/noise-removal:v1.0-allinone
```

## 🌐 Production Deployment

```yaml
version: '3.8'
services:
  noise-removal:
    image: neosun/noise-removal:v1.0-allinone
    container_name: noise-removal
    ports:
      - "5080:5080"
    environment:
      - CUSTOM_DOMAIN=your-domain.com
      - USE_HTTPS=true
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped
```

## 📈 Performance

- **Processing Speed**: ~10x faster than real-time on L40S GPU
- **Model Load Time**: <5 seconds (pre-loaded in image)
- **Memory Usage**: ~2GB VRAM per request

## 🐛 Troubleshooting

### Model not loading
```bash
# Check logs
docker logs noise-removal

# Verify GPU access
docker exec noise-removal nvidia-smi
```

### Port already in use
```bash
# Use different port
docker run -d --gpus all -p 5081:5080 neosun/noise-removal:v1.0-allinone
```

## 📄 License

MIT License

## 🙏 Credits

- DeepFilterNet by ResembleAI
- Built with Flask, PyTorch, and CUDA

## 📞 Support

- GitHub Issues: [Report a bug](https://github.com/neosun100/noise-removal/issues)
- Documentation: https://noise.aws.xin/docs/

---

**Image**: `neosun/noise-removal:v1.0-allinone`  
**Digest**: `sha256:895034c9f72437b7984333df61dd50a2162887477ca71c410fb23cec49a33fba`  
**Created**: 2025-12-07
