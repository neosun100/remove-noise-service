# ✅ 部署检查清单

## 📋 部署前检查

### 1. 系统要求
- [ ] Ubuntu/Debian Linux 系统
- [ ] Docker 已安装 (`docker --version`)
- [ ] Docker Compose 已安装 (`docker-compose --version`)
- [ ] NVIDIA 驱动已安装 (`nvidia-smi`)
- [ ] NVIDIA Docker Runtime 已安装 (`docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`)

### 2. 硬件要求
- [ ] 至少 1 张 NVIDIA GPU
- [ ] GPU 显存 ≥ 4GB
- [ ] 系统内存 ≥ 8GB
- [ ] 可用磁盘空间 ≥ 10GB

### 3. 网络要求
- [ ] 端口 5080 未被占用 (`netstat -tuln | grep 5080`)
- [ ] 可访问 Docker Hub
- [ ] 可访问 PyPI (pip)
- [ ] 可访问 ModelScope（首次下载模型）

---

## 🚀 部署步骤

### Step 1: 准备项目
```bash
cd /home/neo/upload/remove-noise-service
ls -la  # 确认所有文件存在
```

**检查文件：**
- [ ] Dockerfile
- [ ] docker-compose.yml
- [ ] .env.example
- [ ] start.sh (可执行)
- [ ] api_enhanced.py
- [ ] gpu_manager.py
- [ ] ui_template.html
- [ ] requirements.txt

### Step 2: 配置环境
```bash
# 复制配置文件
cp .env.example .env

# 根据需要编辑配置
nano .env
```

**配置项检查：**
- [ ] PORT 设置正确
- [ ] GPU_IDLE_TIMEOUT 设置合理
- [ ] CUSTOM_DOMAIN 配置（如需要）
- [ ] USE_HTTPS 设置正确

### Step 3: 启动服务
```bash
./start.sh
```

**启动检查：**
- [ ] GPU 自动选择成功
- [ ] Docker 镜像构建成功
- [ ] 容器启动成功
- [ ] 无错误日志

### Step 4: 验证服务
```bash
# 检查容器状态
docker ps | grep remove-noise-service

# 检查日志
docker-compose logs --tail=50

# 测试健康检查
curl http://localhost:5080/health
```

**服务检查：**
- [ ] 容器运行中
- [ ] 健康检查返回 200
- [ ] 无错误日志

---

## 🧪 功能测试

### 1. Web UI 测试
- [ ] 访问 http://localhost:5080
- [ ] 页面正常加载
- [ ] 可以选择/拖拽文件
- [ ] 上传功能正常
- [ ] 进度显示正常
- [ ] 可以下载结果

### 2. API 测试
```bash
./test_api.sh
```

**API 检查：**
- [ ] 健康检查接口正常
- [ ] GPU 状态接口正常
- [ ] Swagger 文档可访问
- [ ] 异步上传接口正常
- [ ] 状态查询接口正常

### 3. Swagger UI 测试
- [ ] 访问 http://localhost:5080/docs
- [ ] 文档页面正常显示
- [ ] 可以展开接口
- [ ] "Try it out" 功能正常
- [ ] 可以执行测试请求

---

## 🎮 GPU 功能测试

### 1. GPU 选择测试
```bash
# 查看选择的 GPU
cat .env | grep GPU_ID

# 验证容器使用的 GPU
docker exec remove-noise-service nvidia-smi
```

- [ ] 选择了正确的 GPU
- [ ] 容器可以访问 GPU

### 2. 资源管理测试
```bash
# 查看 GPU 状态
curl http://localhost:5080/gpu/status
```

- [ ] 可以查询 GPU 状态
- [ ] model_loaded 状态正确
- [ ] idle_time 正常更新

### 3. 自动释放测试
1. [ ] 上传一个文件处理
2. [ ] 等待超过配置的空闲时间
3. [ ] 查看 GPU 状态，确认模型已释放
4. [ ] 再次上传文件
5. [ ] 确认模型自动重新加载

---

## 🌐 网络配置测试

### 1. 本地访问
- [ ] http://localhost:5080 可访问
- [ ] http://127.0.0.1:5080 可访问
- [ ] http://0.0.0.0:5080 可访问

### 2. 局域网访问
```bash
# 获取服务器 IP
hostname -I
```

- [ ] http://<SERVER_IP>:5080 可访问

### 3. 公网访问（如配置）
- [ ] 自定义域名可访问
- [ ] HTTPS 配置正确
- [ ] 反向代理工作正常

---

## 📊 性能测试

### 1. 单文件处理
- [ ] 小文件 (<5MB) 处理正常
- [ ] 中等文件 (5-20MB) 处理正常
- [ ] 大文件 (20-50MB) 处理正常

### 2. 并发处理
```bash
# 同时上传多个文件
for i in {1..3}; do
  curl -X POST http://localhost:5080/upload_async \
    -F "audio=@test_audio.wav" &
done
wait
```

- [ ] 可以处理并发请求
- [ ] 进度跟踪正确
- [ ] 所有任务都能完成

### 3. 资源监控
```bash
# 监控 GPU 使用
watch -n 1 nvidia-smi

# 监控容器资源
docker stats remove-noise-service
```

- [ ] GPU 利用率合理
- [ ] 内存使用正常
- [ ] CPU 使用正常

---

## 🔒 安全检查

### 1. 文件访问
- [ ] 只能访问 /tmp 目录下的文件
- [ ] 不能访问系统其他目录
- [ ] 路径遍历攻击防护有效

### 2. 文件大小限制
- [ ] 超过 50MB 的文件被拒绝
- [ ] 错误信息友好

### 3. 自动清理
- [ ] 1 小时后文件自动删除
- [ ] 过期任务状态自动清理

---

## 📝 文档检查

- [ ] README_DOCKER.md 完整
- [ ] QUICKSTART.md 清晰
- [ ] API 文档完整
- [ ] 配置说明清楚
- [ ] 故障排查指南有用

---

## 🎯 生产环境额外检查

### 1. 监控和日志
- [ ] 配置日志收集
- [ ] 配置监控告警
- [ ] 配置备份策略

### 2. 高可用
- [ ] 配置健康检查
- [ ] 配置自动重启
- [ ] 配置负载均衡（如需要）

### 3. 安全加固
- [ ] 配置防火墙规则
- [ ] 配置 HTTPS
- [ ] 配置访问控制
- [ ] 定期更新依赖

---

## ✅ 部署完成确认

所有检查项都通过后，部署完成！

**最终验证：**
```bash
# 1. 服务状态
docker ps | grep remove-noise-service

# 2. 健康检查
curl http://localhost:5080/health | jq

# 3. 完整测试
./test_api.sh

# 4. Web UI 测试
# 在浏览器中访问并测试完整流程
```

---

## 📞 问题反馈

如果遇到问题：
1. 查看日志：`docker-compose logs -f`
2. 检查 GPU：`nvidia-smi`
3. 查看容器：`docker ps -a`
4. 重启服务：`docker-compose restart`
5. 提交 Issue

---

**部署日期：** ___________  
**部署人员：** ___________  
**环境类型：** [ ] 开发 [ ] 测试 [ ] 生产  
**备注：** ___________
