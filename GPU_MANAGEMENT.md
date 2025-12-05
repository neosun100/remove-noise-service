# GPU 显存管理功能文档

## 功能概述

本系统实现了智能的 GPU 显存管理，支持 CPU/GPU 之间的模型转移，确保在不使用时完全释放 GPU 资源。

## 核心特性

### 1. 自动显存管理
- ✅ **处理完成后自动卸载**：每次音频处理完成后，自动将模型从 GPU 转移到 CPU
- ✅ **空闲超时转移**：GPU 空闲超过设定时间（默认10分钟）后自动转移到 CPU
- ✅ **懒加载机制**：需要时自动从 CPU 恢复到 GPU，无需手动干预
- ✅ **完全释放显存**：卸载后 GPU 显存占用降至最低（< 1GB）

### 2. 手动控制接口
提供 API 端点手动控制显存管理：

#### 卸载到 CPU
```bash
curl -X POST https://noise.aws.xin/gpu/offload
```
将模型从 GPU 转移到 CPU，保留模型在内存中以便快速恢复。

#### 完全释放
```bash
curl -X POST https://noise.aws.xin/gpu/release
```
完全释放所有资源（GPU + CPU），下次使用需重新加载模型。

#### 查看状态
```bash
curl https://noise.aws.xin/gpu/status
```
返回详细的 GPU 和模型状态信息。

## API 响应示例

### GPU 状态
```json
{
  "device": "cuda",
  "idle_time": 15,
  "idle_timeout": 600,
  "model_on_cpu": true,
  "model_on_gpu": false,
  "will_release_in": 585
}
```

### 卸载响应
```json
{
  "code": 0,
  "msg": "GPU 显存已卸载到 CPU",
  "status": {
    "model_on_gpu": false,
    "model_on_cpu": true
  }
}
```

## 工作流程

### 处理音频时
1. 用户上传音频文件
2. 系统检查模型状态：
   - 如果在 GPU 上：直接使用
   - 如果在 CPU 上：快速转移到 GPU（< 5秒）
   - 如果未加载：从磁盘加载（20-30秒）
3. 处理音频
4. **自动卸载到 CPU**（释放 GPU 显存）

### 空闲时
- 监控线程每 30 秒检查一次
- 空闲超过 10 分钟自动转移到 CPU
- GPU 显存降至基础占用（545 MB）

## 性能指标

| 操作 | 耗时 | GPU 显存占用 |
|------|------|-------------|
| 首次加载模型 | 20-30秒 | ~3-4 GB |
| CPU → GPU 转移 | < 5秒 | ~3-4 GB |
| GPU → CPU 卸载 | < 2秒 | 545 MB |
| 完全释放 | < 1秒 | 545 MB |

## 配置参数

环境变量 `GPU_IDLE_TIMEOUT` 控制空闲超时时间（秒）：
```bash
GPU_IDLE_TIMEOUT=600  # 默认 10 分钟
```

## 验证方法

### 1. 检查 API 状态
```bash
curl https://noise.aws.xin/gpu/status | jq .
```

### 2. 检查实际 GPU 显存
```bash
nvidia-smi --query-gpu=index,memory.used --format=csv
```

### 3. 运行完整测试
```bash
./test_gpu_management.sh
```

## 故障排查

### 问题：GPU 显存未释放
**检查**：
```bash
curl https://noise.aws.xin/gpu/status
```
如果 `model_on_gpu: true`，手动卸载：
```bash
curl -X POST https://noise.aws.xin/gpu/offload
```

### 问题：处理速度慢
**原因**：模型可能在 CPU 上
**解决**：首次处理会自动转移到 GPU，后续处理会更快

## 最佳实践

1. **长时间不使用**：调用 `/gpu/release` 完全释放资源
2. **频繁使用**：让系统自动管理，保持模型在 CPU 缓存
3. **监控显存**：定期检查 `/gpu/status` 确认状态
4. **调整超时**：根据使用频率调整 `GPU_IDLE_TIMEOUT`

## 技术实现

- **模型转移**：使用 PyTorch 的 `.cpu()` 和 `.to(device)` 方法
- **显存清理**：调用 `torch.cuda.empty_cache()` 和 `torch.cuda.synchronize()`
- **垃圾回收**：使用 Python `gc.collect()` 强制回收
- **线程安全**：使用 `threading.Lock()` 保护并发访问

## 更新日志

### v2.1.0 (2025-12-05)
- ✨ 新增 CPU/GPU 模型转移功能
- ✨ 处理完成后自动卸载显存
- ✨ 手动卸载和释放 API
- ✨ 详细的 GPU 状态监控
- 🐛 修复显存泄漏问题
- ⚡ 优化模型加载速度
