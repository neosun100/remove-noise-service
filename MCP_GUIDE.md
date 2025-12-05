# 🔧 MCP 工具使用指南

## 概述

本项目提供了 MCP (Model Context Protocol) 工具，可以通过 MCP 客户端直接调用音频降噪服务。

---

## ✨ 可用工具

### 1. `denoise_path(path: str)`

提交本地音频文件进行降噪处理。

**参数**:
- `path` (string): 音频文件的绝对或相对路径

**返回**:
```json
{
  "task_id": "uuid-here",
  "message": "任务已提交"
}
```

**示例**:
```python
denoise_path("/path/to/audio.mp3")
```

---

### 2. `get_status(task_id: str)`

查询任务的当前状态和进度。

**参数**:
- `task_id` (string): 任务 ID

**返回**:
```json
{
  "task_id": "uuid-here",
  "status": "processing",
  "progress": 75,
  "message": "模型处理中... 75.0%",
  "timestamp": 1234567890.123,
  "detailed_info": {
    "model_progress": 75.0,
    "processing_speed": 2.5,
    "eta_seconds": 10
  }
}
```

**状态值**:
- `processing`: 处理中
- `completed`: 已完成
- `failed`: 失败
- `not_found`: 任务不存在

---

### 3. `get_result(task_id: str)`

获取已完成任务的结果文件路径和下载 URL。

**参数**:
- `task_id` (string): 任务 ID

**返回**:
```json
{
  "status": "completed",
  "result_url": "http://localhost:5080/tmp/audio-remove-noise.wav",
  "output_path": "/path/to/tmp/audio-remove-noise.wav",
  "timestamp": 1234567890.123
}
```

---

## 🚀 配置 MCP 服务器

### 方式一：直接运行

```bash
# 进入项目目录
cd /home/neo/upload/remove-noise-service

# 运行 MCP 服务器
python3 mcp_server.py
```

### 方式二：在 MCP 客户端配置

在你的 MCP 客户端配置文件中添加：

```json
{
  "mcpServers": {
    "remove-noise": {
      "command": "python3",
      "args": ["/home/neo/upload/remove-noise-service/mcp_server.py"],
      "env": {}
    }
  }
}
```

---

## 💡 使用示例

### 完整工作流程

```python
# 1. 提交音频文件
result = denoise_path("/path/to/noisy_audio.mp3")
task_id = result["task_id"]
print(f"任务已提交: {task_id}")

# 2. 查询状态（可以多次调用）
status = get_status(task_id)
print(f"进度: {status['progress']}%")
print(f"状态: {status['message']}")

# 3. 等待完成后获取结果
if status["status"] == "completed":
    result = get_result(task_id)
    print(f"输出文件: {result['output_path']}")
    print(f"下载链接: {result['result_url']}")
```

---

## 🔄 与 Web API 的关系

**重要**: MCP 服务器与 Web API 共享同一个任务队列！

这意味着：
- ✅ 可以通过 MCP 提交任务，通过 Web UI 查看进度
- ✅ 可以通过 Web UI 提交任务，通过 MCP 查询状态
- ✅ 任务 ID 在两个接口之间通用

**示例场景**:
1. 使用 MCP 的 `denoise_path()` 提交文件
2. 在浏览器中访问 `http://localhost:5080/status/<task_id>` 查看进度
3. 或者在 Web UI 上传文件，然后用 MCP 的 `get_status()` 查询

---

## 📝 注意事项

### 文件大小限制

✅ **已移除文件大小限制**

- 可以处理任意大小的音频文件
- 建议单个文件不超过 500MB 以获得最佳性能
- 大文件处理时间会更长

### 支持的格式

- MP3
- WAV
- M4A
- AAC
- FLAC
- 其他常见音频格式

### 处理时间

- 小文件 (<5MB): 约 10-30 秒
- 中等文件 (5-20MB): 约 30-60 秒
- 大文件 (20-100MB): 约 60-180 秒
- 超大文件 (>100MB): 根据文件大小而定

---

## 🧪 测试 MCP 工具

运行测试脚本验证 MCP 工具是否可用：

```bash
python3 test_mcp.py
```

输出示例：
```
🧪 测试 MCP 工具
============================================================

✅ 已注册的工具数量: 3
  • denoise_path: Submit a local audio/video file for denoising
  • get_status: Query the current status/progress of a submitted task
  • get_result: Return the output file path and download URL if the task completed

============================================================
✅ MCP 工具可用！

使用方法:
  1. 在 MCP 客户端配置中添加此服务器
  2. 使用 denoise_path(path) 提交音频文件
  3. 使用 get_status(task_id) 查询处理状态
  4. 使用 get_result(task_id) 获取结果
```

---

## 🔧 故障排查

### 问题：找不到 fastmcp 模块

**解决方案**:
```bash
pip install fastmcp
# 或
pip install -r requirements.txt
```

### 问题：文件路径不存在

**解决方案**:
- 确保提供的是绝对路径
- 或者使用相对于当前工作目录的路径
- 检查文件是否真实存在

### 问题：任务状态为 not_found

**解决方案**:
- 检查 task_id 是否正确
- 任务可能已过期（超过 2 小时自动清理）
- 确保 MCP 服务器和 Web API 使用同一个进程

---

## 📚 更多信息

- **Web UI**: http://localhost:5080
- **API 文档**: http://localhost:5080/docs
- **健康检查**: http://localhost:5080/health
- **GPU 状态**: http://localhost:5080/gpu/status

---

## 🎯 最佳实践

1. **批量处理**: 可以同时提交多个任务，服务会自动排队处理
2. **进度监控**: 定期调用 `get_status()` 查看进度和 ETA
3. **错误处理**: 检查返回的 status 字段，处理 failed 状态
4. **资源管理**: 大文件处理完成后，记得下载并删除临时文件

---

**版本**: v2.0.0  
**更新日期**: 2025-12-05
