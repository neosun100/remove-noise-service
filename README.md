# 🎧 remove-noise-service · 一键音频降噪 + Web UI + API

用一句话概括：把嘈杂音/视频丢进来，得到更清晰的语音输出。内置精简好用的 Web UI、稳定的后端 API、以及生产可用的进度反馈与错误处理。开箱即用，不折腾。🚀

---

## ✨ 你会喜欢它的理由
- 🔊 **效果靠谱**：基于 ModelScope 的 ZipEnhancer 模型，通用场景表现稳定
- ⚡ **交互顺滑**：上传→转码→降噪→下载，全程 UI 实时进度与 ETA 提示
- 🔁 **同步/异步全覆盖**：既能一把梭 `/api`，也能走 `/upload_async` + `/status/<task_id>`
- 🧠 **鲁棒性**：自动修复 NaN/Inf、校验空文件、FFmpeg 超时/报错可见
- 🌐 **生产友好**：`CUSTOM_DOMAIN` + 反代头适配，自动拼装公网下载 URL
- 🧼 **自清洁**：后台定时清理 `tmp/` 与过期任务状态，不拖泥带水
- 🖥️ **零前端成本**：主页 `/` 内置现代化 UI，无需构建工具

> 小提示：仓库内自带轻量模型文件（`models/`），初次体验无需额外下载。

---

## 🧭 它是如何工作的（简图）
```
原始音/视频 ──> FFmpeg 转 16kHz/单声道 ──> ZipEnhancer 降噪 ──> 输出 WAV ──> 可下载 URL/直传
```
- 转码产物命名：`<name>-16kconver.wav`
- 降噪产物命名：`<name>-remove-noise.wav`

---

## 🚀 快速开始（3 步）
1) 安装依赖（建议虚拟环境）
```bash
pip install -r requirements.txt --no-deps
```
2) 可选：安装 CUDA 12.1 版 PyTorch（如需 GPU）
```bash
pip uninstall -y torch torchaudio torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
3) 运行服务
```bash
python api.py
```
- UI：`http://127.0.0.1:5080`
- 健康检查：`/health`

系统依赖：
- Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1`
- CentOS/RHEL: `sudo yum install -y ffmpeg libsndfile`

> 重要：请确保 `datasets==3.0.0`，否则 ModelScope 可能报错。

---

## 🖱️ 用法指南（超简单）
### 1) 直接用网页
- 打开主页 `/` → 拖放或选择文件 → 点击开始 → 等待进度条走完 → 下载结果
- UI 会展示：上传→转码→模型处理→完成，整个链路的进度、剩余时间、文件大小等信息

### 2) 用异步 API（推荐集成）
- 上传并触发任务
```bash
curl -F "audio=@./300.wav" http://127.0.0.1:5080/upload_async
# => {"code":0,"data":{"task_id":"...","status_url":"http://127.0.0.1:5080/status/<task_id>"}}
```
- 轮询状态
```bash
curl http://127.0.0.1:5080/status/<task_id>
# status=completed 后，从 data.result_url 下载
```

### 3) 同步 API（向后兼容）
- 返回下载 URL：
```python
import requests
print(requests.post(
  'http://127.0.0.1:5080/api',
  data={'stream': 0},
  files={'audio': open('./300.wav', 'rb')}
).json())
```
- 直接返回 WAV 内容：
```python
import requests
res = requests.post(
  'http://127.0.0.1:5080/api',
  data={'stream': 1},
  files={'audio': open('./300.wav', 'rb')}
)
with open('denoised.wav', 'wb') as f:
  f.write(res.content)
```

---

## ⚙️ 环境与配置
- `CUSTOM_DOMAIN`：设置公网域名（如 `noise.aws.xin`）用于拼接下载 URL
- `USE_HTTPS`：当反代走 HTTPS 时设 `true`（默认），否则 `false`

示例：
```bash
export CUSTOM_DOMAIN=noise.aws.xin
export USE_HTTPS=true
python api.py
```

---

## 🛡️ 生产部署小抄
- 后端已用 `waitress` 启动，多线程可配置
- 反向代理（Nginx）示例：
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
- 文件清理：`tmp/` 超过 1 小时自动删除，任务状态超 2 小时清理
- 安全性：下载路由校验路径，避免目录穿越

---

## 🧩 常见问题（FAQ）
- 首次调用很慢？
  - 模型首次加载较重，建议保持进程常驻
- ModelScope 报错？
  - 确认 `datasets==3.0.0`；并检查系统是否安装 `libsndfile`、`ffmpeg`
- 下载链接不对？
  - 设置 `CUSTOM_DOMAIN` 和 `USE_HTTPS`，或确认你的反代头已正确传递
- GPU 没生效？
  - 确保安装与你 CUDA 版本匹配的 PyTorch 轮子

---

## 🗂️ 目录结构（精简）
```
.
├── api.py                 # 唯一服务入口（内置 UI + API + 进度 + 清理）
├── mcp_server.py          # MCP 服务器（可选）
├── models/                # 轻量模型文件（开箱即用）
├── tmp/                   # 运行时转码与输出（自动创建/清理）
├── requirements.txt       # 项目依赖（含生产与 MCP 依赖）
└── README.md
```

---

## 🧪 一些实现细节（给工程同学）
- FFmpeg 全程隐藏冗余日志，设有超时保护；错误日志透传到后端日志，便于排障
- 转码/降噪前后均做文件校验，自动将 NaN/Inf 修复为 0，避免模型崩溃
- 模型实例懒加载 + 进程内缓存，避免重复初始化
- UI 使用渐变进度、阶段配色、统计信息（已处理时长、预计剩余、模型进度、处理速度）
- 详细错误信息不会返回给用户，仅在后端日志记录，以平衡安全与易用

---

## 🧵 MCP（可选，放在最后）
若你需要在多代理/自动化工作流中以工具形式调用降噪服务，可使用本仓库提供的 MCP 服务器。

- 启动（建议 Python 3.11）：
```bash
python mcp_server.py
```
- 工具：
  - `denoise_path(path: str)` → 提交本地文件，返回 `task_id`
  - `get_status(task_id: str)` → 查询进度/状态
  - `get_result(task_id: str)` → 返回 `output_path` 与 `result_url`
- 典型编排：`denoise_path` → 轮询 `get_status` → 完成后 `get_result` → 使用任意下载工具按 URL 获取文件

> 说明：当前 MCP 官方 Python 包在 3.12 存在类型注解兼容问题，建议 3.11 运行 MCP 端。

---

## 🏁 许可 & 致谢
- 模型来源：ModelScope `damo/speech_zipenhancer_ans_multiloss_16k_base`
- 若你在生产使用或做了二次开发，欢迎提交 PR / Issue，一起把它打磨得更好！💙
