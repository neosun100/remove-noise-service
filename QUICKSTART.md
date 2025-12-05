# 🚀 快速启动指南

## 一键启动（推荐）

```bash
# 1. 进入项目目录
cd /home/neo/upload/remove-noise-service

# 2. 运行启动脚本
./start.sh
```

就这么简单！脚本会自动：
- ✅ 检测并选择最空闲的 GPU
- ✅ 构建 Docker 镜像
- ✅ 启动服务

## 访问服务

启动成功后，访问：

- 🌐 **Web UI**: http://0.0.0.0:5080
- 📚 **API 文档**: http://0.0.0.0:5080/docs
- 💊 **健康检查**: http://0.0.0.0:5080/health

## 常用命令

```bash
# 查看日志
docker-compose logs -f

# 重启服务
docker-compose restart

# 停止服务
docker-compose down

# 查看 GPU 状态
nvidia-smi

# 测试 API
./test_api.sh
```

## 配置（可选）

如需自定义配置：

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 编辑配置
nano .env

# 3. 重启服务
docker-compose down && ./start.sh
```

## 主要配置项

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| PORT | 服务端口 | 5080 |
| GPU_IDLE_TIMEOUT | GPU 空闲超时（分钟） | 10 |
| CUSTOM_DOMAIN | 自定义域名 | - |
| USE_HTTPS | 使用 HTTPS | true |

## 测试验证

### 方法1: Web UI
1. 打开浏览器访问 http://0.0.0.0:5080
2. 拖拽音频文件到上传区域
3. 等待处理完成
4. 下载结果

### 方法2: API 测试
```bash
# 运行自动化测试
./test_api.sh

# 或手动测试
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"
```

### 方法3: Swagger UI
1. 访问 http://0.0.0.0:5080/docs
2. 展开 `/upload_async` 接口
3. 点击 "Try it out"
4. 上传文件并执行

## 故障排查

### 服务无法启动？
```bash
# 检查 Docker
docker --version
docker-compose --version

# 检查 NVIDIA Docker
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 查看详细日志
docker-compose logs
```

### 端口被占用？
```bash
# 修改端口
echo "PORT=5081" >> .env
docker-compose down
./start.sh
```

### GPU 内存不足？
```bash
# 降低空闲超时时间，更快释放 GPU
echo "GPU_IDLE_TIMEOUT=5" >> .env
docker-compose restart
```

## 下一步

- 📖 查看完整文档: [README_DOCKER.md](README_DOCKER.md)
- 📚 浏览 API 文档: http://0.0.0.0:5080/docs
- 🎮 监控 GPU 状态: http://0.0.0.0:5080/gpu/status

## 需要帮助？

- 查看日志: `docker-compose logs -f`
- 检查健康: `curl http://localhost:5080/health`
- 提交 Issue: GitHub Issues

---

**祝使用愉快！** 🎉
