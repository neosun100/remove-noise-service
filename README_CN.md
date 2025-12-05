# 🎵 音频降噪服务

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> 基于 AI 的音频降噪服务，支持自动 GPU 管理、实时进度跟踪和完整的 API 文档。

---

## ✨ 功能特性

- 🎯 **AI 驱动降噪**：基于 ModelScope ZipEnhancer 模型
- 🎮 **智能 GPU 管理**：自动选择最空闲 GPU，空闲时自动释放
- 🐳 **Docker 就绪**：一键部署，完整 GPU 支持
- 📚 **Swagger API 文档**：交互式 API 文档，访问 `/docs`
- 🌐 **双模式**：现代化 Web UI + RESTful API
- ⚡ **实时进度**：实时进度条，显示 ETA 和处理速度
- 🔄 **自动清理**：临时文件 1 小时后自动清理
- 🌍 **多语言**：英文、简体中文、繁体中文、日文

### 📸 Web UI 预览

![Web UI 截图](https://img.aws.xin/uPic/RJZXJa.png)

*现代化 Web 界面，支持拖拽上传、实时进度跟踪和即时下载*

---

## 🚀 快速开始

### 方式一：Docker（推荐）

```bash
# 克隆仓库
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 启动服务（自动选择最佳 GPU）
./start.sh

# 访问服务
# Web UI: http://0.0.0.0:5080
# API 文档: http://0.0.0.0:5080/docs
```

### 方式二：直接运行

```bash
# 安装依赖
pip install -r requirements.txt --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装系统依赖（Ubuntu/Debian）
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# 启动服务
python api_enhanced.py

# 访问: http://127.0.0.1:5080
```

---

## 📦 安装部署

### 前置要求

- **Docker**: 20.10+（Docker 部署）
- **Docker Compose**: 1.29+
- **NVIDIA Docker**: nvidia-docker2
- **GPU**: NVIDIA GPU，显存 4GB+
- **Python**: 3.10+（直接运行）
- **CUDA**: 12.1+（GPU 加速）

### 系统依赖

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y ffmpeg libsndfile1

# CentOS/RHEL
sudo yum install -y ffmpeg libsndfile
```

### Docker 安装

```bash
# 1. 克隆仓库
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 2. 配置（可选）
cp .env.example .env
nano .env

# 3. 启动服务
./start.sh
```

### 直接安装

```bash
# 1. 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt --no-deps

# 3. 安装 PyTorch（CUDA 版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 启动服务
python api_enhanced.py
```

---

## ⚙️ 配置说明

### 环境变量

从模板创建 `.env` 文件：

```bash
cp .env.example .env
```

| 变量 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `PORT` | 服务端口 | 5080 | 5080 |
| `CUSTOM_DOMAIN` | 自定义域名 | - | noise.example.com |
| `USE_HTTPS` | 使用 HTTPS | true | true/false |
| `GPU_IDLE_TIMEOUT` | GPU 空闲超时（分钟） | 10 | 10 |
| `GPU_ID` | GPU ID（自动选择） | 0 | 0, 1, 2... |

### 配置示例

**开发环境**：
```env
PORT=5080
GPU_IDLE_TIMEOUT=5
USE_HTTPS=false
```

**生产环境**：
```env
PORT=5080
CUSTOM_DOMAIN=noise.example.com
USE_HTTPS=true
GPU_IDLE_TIMEOUT=10
```

---

## 💻 使用方法

### Web UI

1. 打开浏览器：http://0.0.0.0:5080
2. 拖拽音频文件或点击选择
3. 等待处理（显示实时进度）
4. 下载结果

### API 使用

#### 异步处理（推荐）

```bash
# 1. 上传文件
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# 响应：
# {
#   "code": 0,
#   "data": {
#     "task_id": "uuid-here",
#     "status_url": "http://localhost:5080/status/uuid-here"
#   }
# }

# 2. 查询状态
curl http://localhost:5080/status/<task_id>

# 3. 下载结果（从响应中的 result_url）
```

#### 同步处理

```bash
# 返回 URL
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=0"

# 直接下载
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=1" \
  -o output.wav
```

### Python 示例

```python
import requests

# 异步上传
response = requests.post(
    'http://localhost:5080/upload_async',
    files={'audio': open('input.mp3', 'rb')}
)
task_id = response.json()['data']['task_id']

# 查询状态
status = requests.get(f'http://localhost:5080/status/{task_id}')
print(status.json())
```

---

## 📚 API 文档

### 交互式文档

访问 http://0.0.0.0:5080/docs 查看完整的 Swagger 文档。

### 主要接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web UI |
| `/docs` | GET | Swagger API 文档 |
| `/health` | GET | 健康检查 |
| `/gpu/status` | GET | GPU 状态 |
| `/upload_async` | POST | 异步上传 |
| `/status/<task_id>` | GET | 查询任务状态 |
| `/api` | POST | 同步处理 |

---

## 🏗️ 项目结构

```
.
├── api_enhanced.py          # 增强版 API 服务
├── gpu_manager.py           # GPU 资源管理器
├── ui_template.html         # Web UI 模板
├── Dockerfile               # Docker 镜像
├── docker-compose.yml       # Docker Compose 配置
├── start.sh                 # 一键启动脚本
├── test_api.sh             # API 测试脚本
├── Makefile                # 快捷命令
├── requirements.txt        # Python 依赖
├── models/                 # 模型缓存
└── tmp/                    # 临时文件
```

---

## 🛠️ 技术栈

- **后端**：Python 3.10+, Flask, Waitress
- **AI 模型**：ModelScope ZipEnhancer
- **深度学习**：PyTorch, TorchAudio
- **音频处理**：FFmpeg, SoundFile, LibROSA
- **API 文档**：Flasgger (Swagger/OpenAPI)
- **容器化**：Docker, Docker Compose
- **GPU**：CUDA 12.1, NVIDIA Docker

---

## 🔧 常用命令

### 使用 Makefile

```bash
make help      # 显示所有命令
make start     # 启动服务
make stop      # 停止服务
make restart   # 重启服务
make logs      # 查看日志
make test      # 运行测试
make status    # 检查状态
make health    # 健康检查
make gpu       # GPU 状态
```

### 使用 Docker Compose

```bash
docker-compose up -d      # 启动
docker-compose down       # 停止
docker-compose restart    # 重启
docker-compose logs -f    # 查看日志
```

---

## 🧪 测试

### 自动化测试

```bash
# 运行测试脚本
./test_api.sh

# 或使用 make
make test
```

### 手动测试

1. **Web UI**：访问 http://0.0.0.0:5080 并上传文件
2. **API**：访问 http://0.0.0.0:5080/docs 并尝试接口
3. **健康检查**：`curl http://localhost:5080/health`

---

## 🤝 贡献指南

欢迎贡献！请遵循以下步骤：

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📝 更新日志

### v2.0.0 (2025-12-05)
- ✨ 完整 Docker 部署
- ✨ 自动 GPU 选择和管理
- ✨ Swagger API 文档
- ✨ 增强的 Web UI，包含详细说明
- ✨ 实时进度跟踪
- ✨ 一键启动脚本

### v1.0.0
- 🎉 初始版本
- 🎵 基础音频降噪
- 🌐 Web UI
- 📡 API 接口

---

## 📄 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- [ModelScope](https://modelscope.cn/) 提供的 ZipEnhancer 模型
- 所有贡献者和用户

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/remove-noise-service&type=Date)](https://star-history.com/#yourusername/remove-noise-service)

---

## 📱 关注我们

![微信公众号](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

**扫码关注「AI健自习室」获取更多 AI 工具和教程**
