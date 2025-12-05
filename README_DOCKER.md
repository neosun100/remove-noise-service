# 🐳 音频降噪服务 - Docker 完整部署指南

## 📋 目录
- [功能特性](#功能特性)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API 文档](#api-文档)
- [GPU 资源管理](#gpu-资源管理)
- [测试验证](#测试验证)
- [常见问题](#常见问题)

---

## ✨ 功能特性

### 🎯 核心功能
- ✅ **自动 GPU 选择**：启动时自动选择显存占用最少的 GPU
- ✅ **智能资源管理**：空闲 N 分钟后自动释放 GPU，新请求时自动重载
- ✅ **双模式支持**：Web UI + RESTful API，共用一个端口
- ✅ **Swagger 文档**：完整的 API 文档，可通过 `/docs` 访问
- ✅ **实时进度**：UI 实时显示处理进度、ETA、处理速度等
- ✅ **多语言支持**：中文界面，详细的使用说明

### 🔧 技术特性
- 🐳 **完整 Docker 化**：一键启动，环境隔离
- 🎮 **GPU 加速**：支持 NVIDIA CUDA 12.1
- 📊 **资源监控**：实时查看 GPU 状态和模型加载情况
- 🔄 **自动清理**：临时文件和过期任务自动清理
- 🌐 **反向代理友好**：支持自定义域名和 HTTPS

---

## 🚀 快速开始

### 前置要求
- Docker 和 Docker Compose
- NVIDIA Docker Runtime（nvidia-docker）
- 至少一张 NVIDIA GPU

### 1. 克隆项目
```bash
cd /home/neo/upload/remove-noise-service
```

### 2. 配置环境变量
```bash
# 复制示例配置
cp .env.example .env

# 编辑配置（可选）
nano .env
```

### 3. 一键启动
```bash
./start.sh
```

启动脚本会自动：
- ✅ 检测所有可用 GPU
- ✅ 选择显存占用最少的 GPU
- ✅ 构建 Docker 镜像
- ✅ 启动服务容器

### 4. 访问服务
- 🌐 **Web UI**: http://0.0.0.0:5080
- 📚 **API 文档**: http://0.0.0.0:5080/docs
- 💊 **健康检查**: http://0.0.0.0:5080/health
- 🎮 **GPU 状态**: http://0.0.0.0:5080/gpu/status

---

## ⚙️ 配置说明

### 环境变量（.env 文件）

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `PORT` | 服务端口 | 5080 | 5080 |
| `CUSTOM_DOMAIN` | 自定义域名 | - | noise.aws.xin |
| `USE_HTTPS` | 是否使用 HTTPS | true | true/false |
| `GPU_IDLE_TIMEOUT` | GPU 空闲超时（分钟） | 10 | 10 |
| `GPU_ID` | GPU ID（自动选择） | 0 | 0, 1, 2... |

### 示例配置

#### 本地开发
```env
PORT=5080
GPU_IDLE_TIMEOUT=5
USE_HTTPS=false
```

#### 生产环境
```env
PORT=5080
CUSTOM_DOMAIN=noise.example.com
USE_HTTPS=true
GPU_IDLE_TIMEOUT=10
```

---

## 📚 API 文档

### Swagger UI
访问 `/docs` 查看完整的交互式 API 文档。

### 主要接口

#### 1. 异步上传处理（推荐）
```bash
# 上传文件
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# 响应
{
  "code": 0,
  "msg": "文件上传成功，正在处理中",
  "data": {
    "task_id": "uuid-here",
    "status_url": "http://localhost:5080/status/uuid-here",
    "estimated_time": "30-90秒"
  }
}

# 查询状态
curl http://localhost:5080/status/uuid-here

# 响应（处理中）
{
  "code": 0,
  "data": {
    "task_id": "uuid-here",
    "status": "processing",
    "progress": 75,
    "message": "模型处理中... 75.0%",
    "detailed_info": {
      "model_progress": 75.0,
      "processing_speed": 2.5,
      "eta_seconds": 10
    }
  }
}

# 响应（完成）
{
  "code": 0,
  "data": {
    "task_id": "uuid-here",
    "status": "completed",
    "progress": 100,
    "message": "降噪处理完成！",
    "result_url": "http://localhost:5080/tmp/audio-remove-noise.wav"
  }
}
```

#### 2. 同步处理（向后兼容）
```bash
# 返回下载 URL
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=0"

# 直接返回音频文件
curl -X POST http://localhost:5080/api \
  -F "audio=@your_audio.mp3" \
  -F "stream=1" \
  -o output.wav
```

#### 3. 系统接口
```bash
# 健康检查
curl http://localhost:5080/health

# GPU 状态
curl http://localhost:5080/gpu/status
```

---

## 🎮 GPU 资源管理

### 自动资源管理
服务内置智能 GPU 资源管理器：

1. **懒加载**：首次请求时才加载模型
2. **自动释放**：空闲超过配置时间后自动释放 GPU 内存
3. **自动重载**：新请求到来时自动重新加载模型
4. **实时监控**：通过 `/gpu/status` 查看当前状态

### GPU 状态查询
```bash
curl http://localhost:5080/gpu/status
```

响应示例：
```json
{
  "model_loaded": true,
  "idle_time": 120,
  "idle_timeout": 600,
  "will_release_in": 480
}
```

### 配置超时时间
在 `.env` 文件中设置：
```env
GPU_IDLE_TIMEOUT=10  # 10 分钟
```

或在 Web UI 中实时查看和调整。

---

## 🧪 测试验证

### 自动化测试
```bash
# 运行测试脚本
./test_api.sh

# 指定服务地址
./test_api.sh http://your-server:5080
```

测试脚本会验证：
- ✅ 健康检查接口
- ✅ GPU 状态接口
- ✅ Swagger 文档可访问性
- ✅ 异步上传和处理流程

### 手动测试

#### 1. Web UI 测试
1. 访问 http://localhost:5080
2. 拖拽或选择音频文件
3. 观察实时进度
4. 下载处理结果

#### 2. API 测试
```bash
# 准备测试文件
# 将任意音频文件命名为 test_audio.wav

# 上传测试
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@test_audio.wav"

# 查看 Swagger 文档
open http://localhost:5080/docs
```

---

## 🔧 常见问题

### Q1: 如何查看服务日志？
```bash
docker-compose logs -f
```

### Q2: 如何重启服务？
```bash
docker-compose restart
```

### Q3: 如何停止服务？
```bash
docker-compose down
```

### Q4: 如何更换 GPU？
```bash
# 方法1: 重新运行启动脚本（自动选择）
./start.sh

# 方法2: 手动指定
echo "GPU_ID=1" >> .env
docker-compose down
docker-compose up -d
```

### Q5: 如何查看 GPU 使用情况？
```bash
# 在宿主机上
nvidia-smi

# 在容器内
docker exec remove-noise-service nvidia-smi
```

### Q6: 服务无法启动？
检查：
1. NVIDIA Docker 是否正确安装
2. GPU 驱动是否正常
3. 端口是否被占用
4. 查看详细日志：`docker-compose logs`

### Q7: 模型下载慢？
首次启动会下载模型文件，可能需要几分钟。模型会缓存在 `./models` 目录。

### Q8: 如何配置反向代理？
Nginx 示例：
```nginx
server {
    listen 80;
    server_name noise.example.com;

    location / {
        proxy_pass http://127.0.0.1:5080;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Forwarded-Host $host;
        
        # WebSocket 支持（如需要）
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 📊 性能优化

### 建议配置
- **GPU 内存**: 至少 4GB
- **系统内存**: 至少 8GB
- **CPU**: 4 核心以上
- **存储**: SSD 推荐

### 并发处理
服务使用线程池处理多个请求，默认 4 个工作线程。可在代码中调整：
```python
executor = ThreadPoolExecutor(max_workers=8)  # 增加到 8
```

---

## 🛠️ 开发指南

### 项目结构
```
.
├── api_enhanced.py          # 增强版 API 服务
├── gpu_manager.py           # GPU 资源管理器
├── ui_template.html         # Web UI 模板
├── Dockerfile               # Docker 镜像定义
├── docker-compose.yml       # Docker Compose 配置
├── start.sh                 # 一键启动脚本
├── test_api.sh             # API 测试脚本
├── .env.example            # 环境变量示例
├── requirements.txt        # Python 依赖
├── models/                 # 模型缓存目录
└── tmp/                    # 临时文件目录
```

### 本地开发
```bash
# 安装依赖
pip install -r requirements.txt

# 运行服务
python api_enhanced.py
```

---

## 📝 更新日志

### v2.0.0 (2025-12-05)
- ✨ 完整 Docker 化部署
- ✨ 自动 GPU 选择和资源管理
- ✨ Swagger API 文档
- ✨ 优化的 Web UI
- ✨ 实时进度和详细统计
- ✨ 一键启动脚本

---

## 📄 许可证
本项目基于原 remove-noise-service 项目，遵循相同的许可协议。

---

## 🙏 致谢
- ModelScope ZipEnhancer 模型
- 原项目作者和贡献者

---

## 📞 支持
如有问题或建议，请提交 Issue 或 Pull Request。
