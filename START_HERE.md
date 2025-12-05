# 🎯 从这里开始

欢迎使用音频降噪服务 Docker 版！

---

## ⚡ 3 步快速启动

### 1️⃣ 启动服务
```bash
./start.sh
```

### 2️⃣ 访问服务
打开浏览器访问：http://0.0.0.0:5080

### 3️⃣ 开始使用
- 拖拽音频文件到上传区域
- 等待处理完成
- 下载结果

就这么简单！🎉

---

## 📚 更多资源

### 快速参考
- **快速开始**: [QUICKSTART.md](QUICKSTART.md)
- **完整文档**: [README_DOCKER.md](README_DOCKER.md)
- **部署清单**: [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)
- **项目总结**: [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

### 在线资源
- **Web UI**: http://0.0.0.0:5080
- **API 文档**: http://0.0.0.0:5080/docs
- **健康检查**: http://0.0.0.0:5080/health
- **GPU 状态**: http://0.0.0.0:5080/gpu/status

---

## 🔧 常用命令

### 使用 Makefile（推荐）
```bash
make help      # 查看所有命令
make start     # 启动服务
make stop      # 停止服务
make restart   # 重启服务
make logs      # 查看日志
make test      # 运行测试
make status    # 查看状态
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

### 使用脚本
```bash
./start.sh     # 启动（自动选择 GPU）
./test_api.sh  # 测试 API
```

---

## 🧪 测试验证

### 自动化测试
```bash
make test
# 或
./test_api.sh
```

### 手动测试
1. **Web UI**: 访问 http://0.0.0.0:5080，上传文件测试
2. **API**: 访问 http://0.0.0.0:5080/docs，使用 Swagger 测试
3. **健康检查**: `curl http://localhost:5080/health`

---

## ⚙️ 配置（可选）

如需自定义配置：

```bash
# 1. 复制配置模板
cp .env.example .env

# 2. 编辑配置
nano .env

# 3. 重启服务
make restart
```

### 主要配置项
- `PORT`: 服务端口（默认 5080）
- `GPU_IDLE_TIMEOUT`: GPU 空闲超时分钟数（默认 10）
- `CUSTOM_DOMAIN`: 自定义域名（可选）
- `USE_HTTPS`: 是否使用 HTTPS（默认 true）

---

## 🎮 GPU 管理

### 查看 GPU 状态
```bash
make gpu
# 或
curl http://localhost:5080/gpu/status
```

### GPU 自动管理
服务会自动：
- ✅ 启动时选择最空闲的 GPU
- ✅ 首次请求时加载模型
- ✅ 空闲超时后释放 GPU
- ✅ 新请求时重新加载模型

---

## 📊 监控

### 查看服务状态
```bash
make status
```

### 查看日志
```bash
make logs
```

### 查看资源使用
```bash
docker stats remove-noise-service
```

---

## 🐛 故障排查

### 服务无法启动？
```bash
# 1. 检查 Docker
docker --version
docker-compose --version

# 2. 检查 NVIDIA Docker
nvidia-smi

# 3. 查看日志
make logs
```

### 端口被占用？
```bash
# 修改端口
echo "PORT=5081" >> .env
make restart
```

### GPU 问题？
```bash
# 查看 GPU
nvidia-smi

# 查看容器 GPU
docker exec remove-noise-service nvidia-smi
```

---

## 📖 API 使用

### 异步处理（推荐）
```bash
# 1. 上传文件
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# 2. 查询状态
curl http://localhost:5080/status/<task_id>

# 3. 下载结果（从返回的 result_url）
```

### 同步处理
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

### 更多示例
访问 Swagger 文档查看完整 API：http://0.0.0.0:5080/docs

---

## 🎯 功能特性

### ✨ 核心功能
- 🎵 AI 驱动的音频降噪
- 🚀 自动 GPU 选择和管理
- 🌐 现代化 Web UI
- 📚 完整的 API 文档
- ⚡ 实时进度显示
- 🔄 自动资源清理

### 🔧 技术特性
- 🐳 Docker 容器化
- 🎮 CUDA GPU 加速
- 📊 实时监控
- 🔒 安全防护
- 📝 详细日志

---

## 💡 使用技巧

### 1. 批量处理
可以同时上传多个文件，服务会自动排队处理。

### 2. 监控进度
Web UI 会实时显示：
- 处理进度百分比
- 预计剩余时间
- 处理速度
- 模型进度

### 3. GPU 优化
- 调整 `GPU_IDLE_TIMEOUT` 来平衡性能和资源使用
- 较短的超时时间：更快释放 GPU，适合共享环境
- 较长的超时时间：减少重载次数，适合频繁使用

### 4. 性能优化
- 使用 SSD 存储可提升处理速度
- 增加系统内存可支持更大文件
- 使用更强的 GPU 可加快处理

---

## 📞 获取帮助

### 文档
- [快速开始](QUICKSTART.md)
- [完整文档](README_DOCKER.md)
- [部署清单](DEPLOYMENT_CHECKLIST.md)
- [项目总结](PROJECT_SUMMARY.md)

### 在线资源
- Swagger API 文档: http://0.0.0.0:5080/docs
- 健康检查: http://0.0.0.0:5080/health
- GPU 状态: http://0.0.0.0:5080/gpu/status

### 命令帮助
```bash
make help           # Makefile 命令帮助
./start.sh --help   # 启动脚本帮助（如有）
```

---

## ✅ 验证部署

运行验证脚本确保一切正常：
```bash
./verify_deployment.sh
```

---

## 🎉 开始使用

现在你已经准备好了！

1. **启动服务**: `./start.sh`
2. **打开浏览器**: http://0.0.0.0:5080
3. **上传音频**: 拖拽或选择文件
4. **等待处理**: 观察实时进度
5. **下载结果**: 点击下载按钮

享受你的降噪之旅！🎵

---

**需要更多帮助？** 查看 [README_DOCKER.md](README_DOCKER.md) 获取完整文档。
