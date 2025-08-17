## remove-noise-service

音频降噪服务（Flask + ModelScope）与 MCP 接口。上传音/视频后自动用 FFmpeg 转为 16kHz 单声道 WAV，再通过 ModelScope 的 ZipEnhancer 模型进行降噪，最终返回可下载链接或直接返回音频流。同时提供 MCP 工具，方便在多代理/自动化工作流中编排调用。

### 功能特性
- 异步处理 + 实时进度：`/upload_async` + `/status/<task_id>`，UI 显示进度、预计剩余时间、处理统计
- 同步接口兼容：`/api` 支持一次请求完成（可选择返回下载 URL 或直接返回 WAV 数据）
- 生产化增强：懒加载 + 模型缓存、文件有效性校验、定时清理 `tmp/`、健康检查 `/health`
- 部署友好：支持 `CUSTOM_DOMAIN` 和常见反向代理头 `X-Forwarded-*`，自动生成正确的下载 URL
- 开箱即用 UI：主页 `/` 内置现代化前端，无需额外构建
- MCP 工具：`mcp_server.py` 暴露 `denoise_path` / `get_status` / `get_result` 三个工具，便于工作流集成

### 系统要求
- Python 3.10 ~ 3.12（MCP 官方 Python 包目前在 3.12 上有类型注解兼容性问题，建议 3.11 运行 MCP 端）
- 系统依赖：
  - Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1`
  - CentOS/RHEL: `sudo yum install -y ffmpeg libsndfile`
- 可选：NVIDIA GPU + 对应 CUDA 版本 PyTorch（本仓库 README 以 CUDA 12.1 为例）

### 安装
```bash
# 克隆并进入
# git clone https://github.com/neosun100/remove-noise-service.git
# cd remove-noise-service

# 建议使用虚拟环境
# python -m venv .venv && source .venv/bin/activate

# 安装依赖（避免拉入极大依赖树，可用 --no-deps）
pip install -r requirements.txt --no-deps

# 如需 GPU/CUDA 12.1（可选）
pip uninstall -y torch torchaudio torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 重要：请确保 datasets 版本为 3.0.0，避免 modelscope 兼容问题
pip list | grep datasets  # 期望输出 3.0.0
```

### 运行
```bash
python api.py
```
- 本地 UI：`http://127.0.0.1:5080`
- 健康检查：`http://127.0.0.1:5080/health`

### 环境变量
- `CUSTOM_DOMAIN`：自定义域名（反向代理/公网场景），用于拼装下载 URL，比如：`noise.aws.xin`
- `USE_HTTPS`：当使用自定义域名且走 HTTPS 时设为 `true`（默认 `true`）；否则 `false`

示例：
```bash
export CUSTOM_DOMAIN=noise.aws.xin
export USE_HTTPS=true
python api.py
```

### 接口与交互
- 主页 `/`：上传文件 → 异步处理 → 实时进度/ETA → 下载链接
- 异步 API：
  - `POST /upload_async`（表单字段 `audio` 为文件） → 返回 `task_id`、`status_url`、预计时间
  - `GET /status/<task_id>` → 返回 `status`（processing/completed/failed）、`progress`、`message`、`detailed_info`、`result_url`
- 同步 API（向后兼容）：
  - `GET/POST /api`
    - 参数：
      - 表单或查询 `url`（指向本地 16k WAV 文件路径）或直接上传 `audio` 文件
      - `stream`：`0`（默认，返回 JSON 含下载 URL），`1`（直接返回 WAV 二进制）
- 文件下载：`GET /tmp/<filename>`
- 健康检查：`GET /health`

### 示例
- 异步（curl）：
```bash
# 1) 上传并触发降噪
curl -F "audio=@./300.wav" http://127.0.0.1:5080/upload_async
# 返回: {"code":0,"data":{"task_id":"...","status_url":"http://127.0.0.1:5080/status/<task_id>"}}

# 2) 轮询状态
curl http://127.0.0.1:5080/status/<task_id>
# 当 status=completed 时，从 data.result_url 下载
```

- 同步（Python）：
```python
import requests

# 返回下载 URL
res = requests.post('http://127.0.0.1:5080/api', data={'stream': 0}, files={'audio': open('./300.wav', 'rb')})
print(res.json())

# 直接返回 WAV 内容
res = requests.post('http://127.0.0.1:5080/api', data={'stream': 1}, files={'audio': open('./300.wav', 'rb')})
with open('denoised.wav', 'wb') as f:
    f.write(res.content)
```

### MCP 接口
通过 `mcp_server.py` 提供 MCP 工具，便于在多代理/自动化编排中使用。

- 启动 MCP（建议 Python 3.11）：
```bash
python mcp_server.py
```
- 工具清单：
  - `denoise_path(path: str)`：提交本地文件路径，返回 `task_id`
  - `get_status(task_id: str)`：查询任务状态/进度/信息
  - `get_result(task_id: str)`：当任务完成时返回 `output_path` 和 `result_url`
- 典型编排：
  1) `denoise_path` → 获取 `task_id`
  2) 定时 `get_status` → 获取 `progress/status`
  3) 完成后 `get_result` → 获取 `output_path` 与 `result_url`
  4) 用当前 MCP 客户端或其他 MCP 的下载工具按 `result_url` 获取文件

> 说明：MCP 进程与 Web 进程可以独立运行，各自维护内存任务表；若需共享，可将两者整合在同一进程或引入持久化方案。

### 工作流与实现要点
- 转码：接收原始音/视频，用 FFmpeg 转为 `16kHz/Mono` WAV，命名为 `*-16kconver.wav`
- 降噪：调用 ModelScope ZipEnhancer，生成 `*-remove-noise.wav`
- 进度：解析模型输出中的进度（如 `current_idx: ... 12.3%`），映射为 UI 的 80%-99% 阶段进展，并估算 ETA/处理速度
- 缓存：模型实例在进程内缓存，首次加载后复用
- 清理：后台线程定时清理 `tmp/` 旧文件（>1h）与过期任务状态（>2h）
- 安全：下载接口做路径校验，避免越权访问

### 反向代理（Nginx）参考
```nginx
server {
  listen 80;
  server_name noise.aws.xin;

  location / {
    proxy_pass http://127.0.0.1:5080;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_set_header X-Forwarded-Host $host;
  }
}
```
配置好域名与 TLS 后，设置 `CUSTOM_DOMAIN` 与 `USE_HTTPS`，服务会在返回中使用域名拼装 `result_url`。

### 目录结构（简）
- `api.py`：唯一服务入口（内置 UI + API）
- `mcp_server.py`：MCP server（denoise/status/result 工具）
- `requirements.txt`：项目依赖
- `.gitignore`：忽略 `hf_cache/`、`models/`、`tmp/` 等本地缓存与产物
- `tmp/`：运行时转码与输出（自动创建/清理）
- `models/`：模型缓存（受 `MODELSCOPE_CACHE`、`HF_HOME` 指定）
- `static/`、`templates/`：历史资源（当前入口未直接使用）

### 常见问题（FAQ）
- 首次请求较慢：首次加载模型较耗时，建议进程常驻
- `datasets` 版本：请使用 3.0.0；否则 modelscope 可能报错
- 依赖缺失：若报错 `libsndfile` 或 `ffmpeg`，请先安装系统依赖
- MCP on Python 3.12：当前 `mcp` 官方包在 3.12 有类型注解兼容问题，建议使用 3.11 运行 MCP 端
