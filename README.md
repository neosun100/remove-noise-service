# 🎵 Audio Noise Removal Service

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> AI-powered audio noise removal service with automatic GPU management, real-time progress tracking, and comprehensive API documentation.

---

## ✨ Features

- 🎯 **AI-Powered Denoising**: Based on ModelScope ZipEnhancer model
- 🎮 **Smart GPU Management**: Auto-select least busy GPU, auto-release on idle
- 🐳 **Docker Ready**: One-command deployment with full GPU support
- 📚 **Swagger API Docs**: Interactive API documentation at `/docs`
- 🌐 **Dual Mode**: Modern Web UI + RESTful API
- ⚡ **Real-time Progress**: Live progress bar with ETA and processing speed
- 🔄 **Auto Cleanup**: Temporary files cleaned after 1 hour
- 🌍 **Multi-language**: English, Chinese (Simplified/Traditional), Japanese

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Clone repository
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# Start service (auto-selects best GPU)
./start.sh

# Access service
# Web UI: http://0.0.0.0:5080
# API Docs: http://0.0.0.0:5080/docs
```

### Option 2: Direct Run

```bash
# Install dependencies
pip install -r requirements.txt --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# Start service
python api_enhanced.py

# Access: http://127.0.0.1:5080
```

---

## 📦 Installation

### Prerequisites

- **Docker**: 20.10+ (for Docker deployment)
- **Docker Compose**: 1.29+
- **NVIDIA Docker**: nvidia-docker2
- **GPU**: NVIDIA GPU with 4GB+ VRAM
- **Python**: 3.10+ (for direct run)
- **CUDA**: 12.1+ (for GPU acceleration)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1

# CentOS/RHEL
sudo yum install -y ffmpeg libsndfile
```

### Docker Installation

```bash
# 1. Clone repository
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 2. Configure (optional)
cp .env.example .env
nano .env

# 3. Start service
./start.sh
```

### Direct Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt --no-deps

# 3. Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. Start service
python api_enhanced.py
```

---

## ⚙️ Configuration

### Environment Variables

Create `.env` file from template:

```bash
cp .env.example .env
```

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `PORT` | Service port | 5080 | 5080 |
| `CUSTOM_DOMAIN` | Custom domain | - | noise.example.com |
| `USE_HTTPS` | Use HTTPS | true | true/false |
| `GPU_IDLE_TIMEOUT` | GPU idle timeout (minutes) | 10 | 10 |
| `GPU_ID` | GPU ID (auto-selected) | 0 | 0, 1, 2... |

### Example Configurations

**Development**:
```env
PORT=5080
GPU_IDLE_TIMEOUT=5
USE_HTTPS=false
```

**Production**:
```env
PORT=5080
CUSTOM_DOMAIN=noise.example.com
USE_HTTPS=true
GPU_IDLE_TIMEOUT=10
```

---

## 💻 Usage

### Web UI

1. Open browser: http://0.0.0.0:5080
2. Drag & drop audio file or click to select
3. Wait for processing (real-time progress shown)
4. Download result

### API Usage

#### Async Processing (Recommended)

```bash
# 1. Upload file
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# Response:
# {
#   "code": 0,
#   "data": {
#     "task_id": "uuid-here",
#     "status_url": "http://localhost:5080/status/uuid-here"
#   }
# }

# 2. Check status
curl http://localhost:5080/status/<task_id>

# 3. Download result (from result_url in response)
```

#### Sync Processing

```bash
# Return URL
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=0"

# Direct download
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=1" \
  -o output.wav
```

### Python Example

```python
import requests

# Async upload
response = requests.post(
    'http://localhost:5080/upload_async',
    files={'audio': open('input.mp3', 'rb')}
)
task_id = response.json()['data']['task_id']

# Check status
status = requests.get(f'http://localhost:5080/status/{task_id}')
print(status.json())
```

---

## 📚 API Documentation

### Interactive Docs

Visit http://0.0.0.0:5080/docs for full Swagger documentation.

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/docs` | GET | Swagger API docs |
| `/health` | GET | Health check |
| `/gpu/status` | GET | GPU status |
| `/upload_async` | POST | Async upload |
| `/status/<task_id>` | GET | Check task status |
| `/api` | POST | Sync processing |

---

## 🏗️ Project Structure

```
.
├── api_enhanced.py          # Enhanced API service
├── gpu_manager.py           # GPU resource manager
├── ui_template.html         # Web UI template
├── Dockerfile               # Docker image
├── docker-compose.yml       # Docker Compose config
├── start.sh                 # One-click startup script
├── test_api.sh             # API test script
├── Makefile                # Quick commands
├── requirements.txt        # Python dependencies
├── models/                 # Model cache
└── tmp/                    # Temporary files
```

---

## 🛠️ Tech Stack

- **Backend**: Python 3.10+, Flask, Waitress
- **AI Model**: ModelScope ZipEnhancer
- **Deep Learning**: PyTorch, TorchAudio
- **Audio Processing**: FFmpeg, SoundFile, LibROSA
- **API Docs**: Flasgger (Swagger/OpenAPI)
- **Containerization**: Docker, Docker Compose
- **GPU**: CUDA 12.1, NVIDIA Docker

---

## 🔧 Commands

### Using Makefile

```bash
make help      # Show all commands
make start     # Start service
make stop      # Stop service
make restart   # Restart service
make logs      # View logs
make test      # Run tests
make status    # Check status
make health    # Health check
make gpu       # GPU status
```

### Using Docker Compose

```bash
docker-compose up -d      # Start
docker-compose down       # Stop
docker-compose restart    # Restart
docker-compose logs -f    # View logs
```

---

## 🧪 Testing

### Automated Tests

```bash
# Run test script
./test_api.sh

# Or use make
make test
```

### Manual Testing

1. **Web UI**: Visit http://0.0.0.0:5080 and upload a file
2. **API**: Visit http://0.0.0.0:5080/docs and try endpoints
3. **Health**: `curl http://localhost:5080/health`

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 Changelog

### v2.0.0 (2025-12-05)
- ✨ Complete Docker deployment
- ✨ Auto GPU selection and management
- ✨ Swagger API documentation
- ✨ Enhanced Web UI with detailed instructions
- ✨ Real-time progress tracking
- ✨ One-click startup script

### v1.0.0
- 🎉 Initial release
- 🎵 Basic audio denoising
- 🌐 Web UI
- 📡 API endpoints

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- [ModelScope](https://modelscope.cn/) for the ZipEnhancer model
- All contributors and users

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/remove-noise-service&type=Date)](https://star-history.com/#yourusername/remove-noise-service)

---

## 📱 Follow Us

![WeChat Official Account](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

**Scan to follow "AI健自习室" for more AI tools and tutorials**
