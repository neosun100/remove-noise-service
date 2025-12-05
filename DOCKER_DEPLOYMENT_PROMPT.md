# 项目完整 Docker 化部署 Prompt

> 适用于所有 GPU 服务的标准化部署模板

---

## 📋 背景说明

当前机器有多张 GPU，已配置好 nvidia-docker，有其他 GPU 容器在运行。

**重要要求**：
- ✅ Docker 服务对所有 IP 开放访问（0.0.0.0）
- ✅ 自动选择显存占用最少的 GPU
- ✅ 实现 GPU 显存智能管理（懒加载 + 即用即卸）

---

## 🎯 前置任务

**首先通读当前项目代码**，理解：
1. 项目功能和工作流程
2. 所有可调参数及其作用
3. 输入输出格式
4. 模型加载方式

---

## ✅ 任务清单

### 1. Docker 化

#### 1.1 创建 Dockerfile
```dockerfile
# 要求：
- 基于 nvidia/cuda 镜像（根据项目需要选择版本）
- 安装所有依赖
- 配置工作目录
- 暴露服务端口
- 设置启动命令
```

#### 1.2 创建 docker-compose.yml
```yaml
# 要求：
- 配置 GPU 支持（runtime: nvidia）
- 映射端口到 0.0.0.0
- 设置环境变量
- 挂载必要的目录
- 配置重启策略
```

#### 1.3 创建 .env.example
```bash
# 必需环境变量：
PORT=5000                    # 服务端口
GPU_IDLE_TIMEOUT=600         # GPU 空闲超时（秒）
NVIDIA_VISIBLE_DEVICES=0     # GPU ID（自动选择）
# ... 其他项目特定参数
```

#### 1.4 创建 start.sh 一键启动脚本
```bash
# 功能：
1. 检查 nvidia-docker 环境
2. 自动选择显存占用最少的 GPU
3. 设置环境变量
4. 启动 docker-compose
5. 显示访问信息
```

**关键代码**：
```bash
# 自动选择最空闲的 GPU
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)
export NVIDIA_VISIBLE_DEVICES=$GPU_ID
```

---

### 2. GPU 显存智能管理 ⭐

> **核心要求**：实现懒加载 + 即用即卸逻辑

#### 2.1 创建 gpu_manager.py

**必需功能**：
```python
class GPUResourceManager:
    def __init__(self, idle_timeout=600):
        """
        Args:
            idle_timeout: 空闲超时时间（秒）
        """
        self.model = None           # GPU 上的模型
        self.model_on_cpu = None    # CPU 缓存
        self.lock = threading.Lock()
    
    def get_model(self, load_func):
        """
        懒加载逻辑：
        1. 如果在 GPU 上 → 直接返回
        2. 如果在 CPU 上 → 快速转移到 GPU（2-5秒）
        3. 如果未加载 → 从磁盘加载（首次）
        """
        pass
    
    def force_offload(self):
        """
        即用即卸：任务完成后立即调用
        将模型从 GPU 转移到 CPU，释放显存
        """
        pass
    
    def force_release(self):
        """
        完全释放：长期不用时调用
        清空 GPU 和 CPU 缓存
        """
        pass
```

**状态转换**：
```
未加载 ──首次请求(20-30s)──→ GPU ──任务完成(2s)──→ CPU ──新请求(2-5s)──→ GPU
  ↑                                                      ↓
  └────────────────────超时/手动释放(1s)─────────────────┘
```

#### 2.2 集成到项目

**标准处理流程**：
```python
# 初始化（全局）
gpu_manager = GPUResourceManager(idle_timeout=600)
gpu_manager.start_monitor()

# 定义模型加载函数
def load_model():
    model = YourModel.from_pretrained('model-name')
    return model

# 处理函数（3步）
def process_task(input_data):
    try:
        # 步骤1: 懒加载
        model = gpu_manager.get_model(load_func=load_model)
        
        # 步骤2: 处理
        result = model(input_data)
        
        # 步骤3: 立即卸载（关键！）
        gpu_manager.force_offload()
        
        return result
        
    except Exception as e:
        # 异常时也要卸载
        gpu_manager.force_offload()
        raise e
```

**验证要求**：
- 空闲时 GPU 显存 < 1GB
- 处理时 GPU 显存正常占用
- 处理完成后显存立即释放

---

### 3. 单 Docker 三模式支持

> **核心要求**：UI + API + MCP 三种访问方式

#### 3.1 模式一：UI 界面

**若原项目无 UI**：
- ✅ 现代化、响应式设计
- ✅ 支持深色模式
- ✅ 自适应宽度
- ✅ 暴露所有可调参数（分组展示）
- ✅ 实时进度显示
- ✅ 多语言支持：
  - 英文（默认）
  - 简体中文
  - 繁体中文
  - 日文

**若原项目有 UI**：
- ✅ 优化现有 UI
- ✅ 补充功能说明
- ✅ 添加中文支持
- ✅ 增加参数说明
- ✅ 方便初学者上手

**UI 必需元素**：
```html
<!-- 参数配置区 -->
<div class="parameters">
    <h3>参数设置</h3>
    <!-- 所有可调参数 -->
</div>

<!-- 文件上传区 -->
<div class="upload">
    <input type="file" />
    <button>处理</button>
</div>

<!-- 进度显示 -->
<div class="progress">
    <div class="progress-bar"></div>
    <span class="status"></span>
</div>

<!-- GPU 状态 -->
<div class="gpu-status">
    <span>GPU: <span id="gpu-status"></span></span>
    <button onclick="offloadGPU()">释放显存</button>
</div>

<!-- 语言切换 -->
<select id="language">
    <option value="en">English</option>
    <option value="zh-CN">简体中文</option>
    <option value="zh-TW">繁體中文</option>
    <option value="ja">日本語</option>
</select>
```

#### 3.2 模式二：API 接口

**根据项目需要选择**：
- RESTful API（推荐用于文件处理）
- WebSocket API（推荐用于实时交互）

**必需端点**：
```python
# 健康检查
GET /health

# GPU 状态
GET /gpu/status

# 手动卸载
POST /gpu/offload

# 完全释放
POST /gpu/release

# 主要功能（根据项目定义）
POST /api/process
GET /api/status/<task_id>
```

**Swagger 文档**：
```python
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app)

@app.route('/api/process', methods=['POST'])
def process():
    """
    处理请求
    ---
    tags:
      - API
    parameters:
      - name: file
        in: formData
        type: file
        required: true
    responses:
      200:
        description: 处理成功
    """
    pass
```

**要求**：
- ✅ Swagger 文档可通过 `/docs` 访问
- ✅ API 与 UI 功能一致
- ✅ 共用一个端口（不同路径）
- ✅ 支持异步处理（长时间任务）

#### 3.3 模式三：MCP 接口 ⭐

> Model Context Protocol - 程序化访问接口

**MCP 服务器配置**：
```python
# mcp_server.py
from fastmcp import FastMCP

mcp = FastMCP("项目名称")

@mcp.tool()
def process_file(file_path: str, **params) -> dict:
    """
    处理文件（MCP 工具）
    
    Args:
        file_path: 文件路径
        **params: 其他参数
    
    Returns:
        处理结果
    """
    try:
        # 使用 GPU 管理器
        model = gpu_manager.get_model(load_func=load_model)
        result = model.process(file_path, **params)
        gpu_manager.force_offload()
        
        return {
            'status': 'success',
            'result': result
        }
    except Exception as e:
        gpu_manager.force_offload()
        return {
            'status': 'error',
            'error': str(e)
        }

@mcp.tool()
def get_gpu_status() -> dict:
    """获取 GPU 状态"""
    return gpu_manager.get_status()

@mcp.tool()
def offload_gpu() -> dict:
    """手动卸载 GPU 显存"""
    gpu_manager.force_offload()
    return {'status': 'offloaded'}

# 启动 MCP 服务器
if __name__ == "__main__":
    mcp.run()
```

**MCP 配置文件**：
```json
{
  "mcpServers": {
    "项目名称": {
      "command": "python",
      "args": ["mcp_server.py"],
      "env": {
        "GPU_IDLE_TIMEOUT": "600"
      }
    }
  }
}
```

**MCP 工具要求**：
- ✅ 提供主要功能的工具函数
- ✅ 提供 GPU 状态查询工具
- ✅ 提供 GPU 管理工具（卸载/释放）
- ✅ 所有工具共享同一个 GPU 管理器
- ✅ 工具函数包含完整的类型注解和文档
- ✅ 错误处理完善

**MCP 与 API 的关系**：
```
┌─────────────────────────────────────────────┐
│           GPU 资源管理器（共享）             │
└─────────────────────────────────────────────┘
         ↓              ↓              ↓
    ┌────────┐    ┌────────┐    ┌────────┐
    │   UI   │    │  API   │    │  MCP   │
    │  Web   │    │  REST  │    │  Tool  │
    └────────┘    └────────┘    └────────┘
```

**MCP 使用示例**：
```python
# 客户端调用（通过 MCP 客户端）
result = await mcp_client.call_tool(
    "process_file",
    {
        "file_path": "/path/to/file",
        "param1": "value1"
    }
)
```

**MCP 文档要求**：
- 创建 `MCP_GUIDE.md` 说明 MCP 工具使用
- 列出所有可用工具及其参数
- 提供使用示例
- 说明与 API 的区别

---

### 4. 资源管理

#### 4.1 自动释放机制

```python
# 监控线程
def _monitor_loop(self):
    while self.running:
        time.sleep(30)  # 每30秒检查
        
        with self.lock:
            if self.model is not None:
                idle_time = time.time() - self.last_use_time
                
                # 超时自动卸载
                if idle_time > self.idle_timeout:
                    logger.info(f"空闲 {idle_time}秒，自动卸载")
                    self._move_to_cpu()
```

#### 4.2 UI 配置

```html
<!-- GPU 超时配置 -->
<div class="gpu-config">
    <label>GPU 空闲超时（分钟）：</label>
    <input type="number" id="gpu-timeout" value="10" min="1" max="60">
    <button onclick="updateTimeout()">更新</button>
</div>

<!-- 当前状态 -->
<div class="gpu-info">
    <p>模型位置: <span id="model-location">未加载</span></p>
    <p>空闲时间: <span id="idle-time">0</span> 秒</p>
    <p>显存占用: <span id="gpu-memory">0</span> MB</p>
</div>
```

---

### 5. 完整文件结构

```
project/
├── Dockerfile                      # Docker 镜像定义
├── docker-compose.yml              # Docker Compose 配置
├── .env.example                    # 环境变量模板
├── start.sh                        # 一键启动脚本
├── requirements.txt                # Python 依赖
├── gpu_manager.py                  # GPU 资源管理器 ⭐
├── api.py                          # API 服务
├── mcp_server.py                   # MCP 服务器 ⭐
├── ui_template.html                # UI 模板
├── README.md                       # 项目说明
├── GPU_MANAGEMENT.md               # GPU 管理文档
├── MCP_GUIDE.md                    # MCP 使用指南 ⭐
└── test_api.sh                     # API 测试脚本
```

---

## 🧪 测试验证清单

### 本地测试

- [ ] Docker 镜像构建成功
- [ ] 容器启动成功
- [ ] 自动选择最空闲 GPU
- [ ] UI 界面可访问（http://0.0.0.0:PORT）
- [ ] API 接口可访问
- [ ] Swagger 文档可访问（/docs）
- [ ] MCP 服务器可连接 ⭐
- [ ] MCP 工具可调用 ⭐
- [ ] 多语言切换正常

### GPU 管理测试

- [ ] 首次请求加载模型（20-30秒）
- [ ] 处理完成后自动卸载（显存 < 1GB）
- [ ] 第二次请求快速恢复（2-5秒）
- [ ] 空闲超时自动转移到 CPU
- [ ] 手动卸载 API 正常
- [ ] 手动释放 API 正常

### 功能测试

- [ ] 文件上传处理正常
- [ ] 参数调整生效
- [ ] 进度显示准确
- [ ] 错误处理正确
- [ ] 结果下载正常

### 验证命令

```bash
# 1. 检查容器状态
docker ps

# 2. 检查 GPU 使用
nvidia-smi

# 3. 测试 API
curl http://localhost:PORT/health
curl http://localhost:PORT/gpu/status

# 4. 测试 Swagger
curl http://localhost:PORT/docs

# 5. 测试处理
curl -X POST http://localhost:PORT/api/process \
  -F "file=@test.txt"

# 6. 测试 MCP（使用 MCP 客户端）
# 方法1: 使用 mcp CLI
mcp call process_file '{"file_path": "/path/to/file"}'

# 方法2: 使用 Python 客户端
python << EOF
from mcp import ClientSession
async with ClientSession() as session:
    result = await session.call_tool("process_file", {
        "file_path": "/path/to/file"
    })
    print(result)
EOF

# 7. 验证显存释放
nvidia-smi --query-gpu=memory.used --format=csv
```

---

## 📊 性能指标要求

| 指标 | 目标值 |
|------|--------|
| 空闲显存占用 | < 1 GB |
| 首次加载时间 | 20-30秒 |
| CPU→GPU 恢复 | 2-5秒 |
| GPU→CPU 卸载 | < 2秒 |
| API 响应时间 | < 100ms（不含处理） |
| UI 加载时间 | < 2秒 |

---

## 📚 参考文档

完成后需创建以下文档：

1. **README.md** - 项目说明
   - 功能介绍
   - 快速开始
   - API 文档
   - MCP 使用 ⭐
   - 配置说明

2. **GPU_MANAGEMENT.md** - GPU 管理说明
   - 工作原理
   - API 端点
   - 性能指标
   - 故障排查

3. **MCP_GUIDE.md** - MCP 使用指南 ⭐
   - MCP 工具列表
   - 参数说明
   - 使用示例
   - 配置方法
   - 与 API 的区别

4. **DEPLOYMENT.md** - 部署指南
   - 环境要求
   - 安装步骤
   - 配置说明
   - 常见问题

---

## 🎯 关键实现要点

### 1. GPU 自动选择

```bash
#!/bin/bash
# start.sh

echo "🔍 检测可用 GPU..."
GPU_ID=$(nvidia-smi --query-gpu=index,memory.used \
         --format=csv,noheader,nounits | \
         sort -t',' -k2 -n | head -1 | cut -d',' -f1)

echo "✅ 选择 GPU: $GPU_ID"
export NVIDIA_VISIBLE_DEVICES=$GPU_ID

docker-compose up -d
```

### 2. 懒加载实现

```python
def get_model(self, load_func):
    with self.lock:
        self.last_use_time = time.time()
        
        # 已在 GPU
        if self.model is not None:
            return self.model
        
        # 在 CPU 缓存（快速恢复）
        if self.model_on_cpu is not None:
            logger.info("从 CPU 恢复到 GPU...")
            self._move_to_gpu()
            return self.model
        
        # 首次加载
        logger.info("首次加载模型...")
        self.model = load_func()
        return self.model
```

### 3. 即用即卸实现

```python
def process_task(data):
    try:
        # 1. 懒加载
        model = gpu_manager.get_model(load_func=load_model)
        
        # 2. 处理
        result = model(data)
        
        # 3. 立即卸载（关键！）
        gpu_manager.force_offload()
        
        return result
    except Exception as e:
        # 异常也要卸载
        gpu_manager.force_offload()
        raise e
```

### 4. API 集成

```python
@app.route('/api/process', methods=['POST'])
def api_process():
    """处理请求（自动管理 GPU）"""
    try:
        data = request.files['file']
        result = process_task(data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/gpu/status')
def gpu_status():
    """GPU 状态"""
    return jsonify(gpu_manager.get_status())

@app.route('/gpu/offload', methods=['POST'])
def gpu_offload():
    """手动卸载"""
    gpu_manager.force_offload()
    return jsonify({'status': 'ok'})
```

---

## ✅ 完成标准

项目完成需满足：

1. **功能完整**
   - ✅ Docker 一键启动
   - ✅ UI、API、MCP 三模式 ⭐
   - ✅ GPU 智能管理
   - ✅ 多语言支持

2. **性能达标**
   - ✅ 空闲显存 < 1GB
   - ✅ 快速恢复 < 5秒
   - ✅ 自动卸载正常

3. **文档齐全**
   - ✅ README 完整
   - ✅ API 文档可访问
   - ✅ MCP 文档完整 ⭐
   - ✅ 部署指南清晰

4. **测试通过**
   - ✅ 本地测试通过
   - ✅ GPU 管理验证
   - ✅ 功能测试通过
   - ✅ MCP 工具验证 ⭐

---

## 🔗 相关资源

- GPU 管理器参考实现：`gpu_manager.py`
- 最佳实践文档：`GPU_MEMORY_BEST_PRACTICES.md`
- 快速参考：`QUICK_REFERENCE.md`
- 测试脚本：`test_gpu_management.sh`

---

**版本**: v1.0  
**更新日期**: 2025-12-05  
**适用范围**: 所有 GPU Docker 服务

---

## 📘 附录：MCP 实现详解

### MCP 是什么？

Model Context Protocol (MCP) 是一个开放协议，用于标准化应用程序如何向 LLM 提供上下文。通过 MCP，可以让 AI 助手（如 Claude Desktop）直接调用你的服务。

### MCP vs API

| 特性 | API | MCP |
|------|-----|-----|
| 访问方式 | HTTP 请求 | 工具调用 |
| 使用场景 | Web/移动应用 | AI 助手集成 |
| 调用方式 | curl/fetch | AI 自动调用 |
| 文档形式 | Swagger | 函数注解 |
| 适用对象 | 开发者 | AI 助手 |

### MCP 完整实现示例

```python
# mcp_server.py
from fastmcp import FastMCP
from gpu_manager import GPUResourceManager
import logging

logger = logging.getLogger(__name__)

# 初始化 GPU 管理器（全局共享）
gpu_manager = GPUResourceManager(idle_timeout=600)
gpu_manager.start_monitor()

# 创建 MCP 服务器
mcp = FastMCP("项目名称")

def load_model():
    """模型加载函数"""
    # 根据项目实现
    pass

@mcp.tool()
def process_file(
    file_path: str,
    param1: str = "default",
    param2: int = 100
) -> dict:
    """
    处理文件的主要功能
    
    Args:
        file_path: 输入文件路径
        param1: 参数1说明
        param2: 参数2说明
    
    Returns:
        包含处理结果的字典
    """
    try:
        logger.info(f"处理文件: {file_path}")
        
        # 1. 懒加载模型
        model = gpu_manager.get_model(load_func=load_model)
        
        # 2. 处理
        result = model.process(file_path, param1=param1, param2=param2)
        
        # 3. 立即卸载
        gpu_manager.force_offload()
        
        return {
            'status': 'success',
            'result': result,
            'file_path': file_path
        }
        
    except Exception as e:
        gpu_manager.force_offload()
        logger.error(f"处理失败: {e}")
        return {
            'status': 'error',
            'error': str(e)
        }

@mcp.tool()
def get_gpu_status() -> dict:
    """
    获取 GPU 状态信息
    
    Returns:
        GPU 状态字典
    """
    return gpu_manager.get_status()

@mcp.tool()
def offload_gpu() -> dict:
    """
    手动卸载 GPU 显存到 CPU
    
    Returns:
        操作结果
    """
    success = gpu_manager.force_offload()
    return {
        'status': 'success' if success else 'no_model',
        'message': 'GPU 显存已卸载' if success else '没有需要卸载的模型'
    }

@mcp.tool()
def release_gpu() -> dict:
    """
    完全释放 GPU 和 CPU 资源
    
    Returns:
        操作结果
    """
    gpu_manager.force_release()
    return {
        'status': 'success',
        'message': '所有资源已释放'
    }

# 启动 MCP 服务器
if __name__ == "__main__":
    mcp.run()
```

### MCP 配置文件

**Claude Desktop 配置** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "项目名称": {
      "command": "docker",
      "args": [
        "exec",
        "-i",
        "container-name",
        "python",
        "mcp_server.py"
      ],
      "env": {
        "GPU_IDLE_TIMEOUT": "600"
      }
    }
  }
}
```

**或者直接运行**:
```json
{
  "mcpServers": {
    "项目名称": {
      "command": "python",
      "args": ["/path/to/mcp_server.py"],
      "env": {
        "GPU_IDLE_TIMEOUT": "600"
      }
    }
  }
}
```

### MCP_GUIDE.md 模板

```markdown
# MCP 使用指南

## 可用工具

### 1. process_file
处理文件的主要功能

**参数**:
- `file_path` (string, 必需): 输入文件路径
- `param1` (string, 可选): 参数1说明，默认 "default"
- `param2` (int, 可选): 参数2说明，默认 100

**返回**:
```json
{
  "status": "success",
  "result": "...",
  "file_path": "/path/to/file"
}
```

**使用示例**:
```
请帮我处理文件 /path/to/input.txt，使用参数1为 "custom"
```

### 2. get_gpu_status
获取 GPU 状态信息

**参数**: 无

**返回**:
```json
{
  "model_on_gpu": false,
  "model_on_cpu": true,
  "idle_time": 120,
  "device": "cuda"
}
```

### 3. offload_gpu
手动卸载 GPU 显存

**参数**: 无

**返回**:
```json
{
  "status": "success",
  "message": "GPU 显存已卸载"
}
```

### 4. release_gpu
完全释放所有资源

**参数**: 无

**返回**:
```json
{
  "status": "success",
  "message": "所有资源已释放"
}
```

## 配置方法

1. 找到 Claude Desktop 配置文件
2. 添加 MCP 服务器配置
3. 重启 Claude Desktop
4. 在对话中使用工具

## 与 API 的区别

- **API**: 需要手动构造 HTTP 请求
- **MCP**: AI 自动调用，自然语言交互

## 注意事项

- 所有工具共享同一个 GPU 管理器
- 处理完成后自动卸载显存
- 支持并发调用（线程安全）
```

### requirements.txt 更新

```txt
# 原有依赖
...

# MCP 支持
fastmcp>=0.4.0
mcp>=1.0.0
```

### Docker 配置更新

**Dockerfile**:
```dockerfile
# 安装 MCP 依赖
RUN pip install fastmcp mcp

# 复制 MCP 服务器
COPY mcp_server.py .
```

### 测试 MCP

```python
# test_mcp.py
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def test_mcp():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"]
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # 测试处理文件
            result = await session.call_tool(
                "process_file",
                {"file_path": "/path/to/test.txt"}
            )
            print("处理结果:", result)
            
            # 测试 GPU 状态
            status = await session.call_tool("get_gpu_status", {})
            print("GPU 状态:", status)

if __name__ == "__main__":
    asyncio.run(test_mcp())
```

### 关键要点

1. **共享 GPU 管理器**: MCP、API、UI 使用同一个 `gpu_manager` 实例
2. **即用即卸**: 每个 MCP 工具调用后都要 `force_offload()`
3. **类型注解**: 所有参数和返回值都要有类型注解
4. **文档字符串**: 详细的 docstring 帮助 AI 理解工具用途
5. **错误处理**: 异常时也要卸载显存

---

**更新**: 2025-12-05 - 添加 MCP 支持
