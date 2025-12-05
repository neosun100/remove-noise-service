# GPU 显存管理 - 快速参考

## 🎯 核心逻辑（3 步）

```python
# 1. 懒加载
model = gpu_manager.get_model(load_func=load_model)

# 2. 处理任务
result = model(data)

# 3. 立即卸载
gpu_manager.force_offload()
```

## 📊 状态转换

```
未加载 ──首次请求(20s)──→ GPU ──任务完成(2s)──→ CPU ──新请求(5s)──→ GPU
  ↑                                                    │
  └──────────────────超时/手动释放(1s)─────────────────┘
```

## 🔧 集成步骤

### 1. 复制管理器
```bash
cp gpu_manager.py your_project/
```

### 2. 初始化
```python
from gpu_manager import GPUResourceManager

gpu_manager = GPUResourceManager(idle_timeout=600)
gpu_manager.start_monitor()
```

### 3. 定义加载函数
```python
def load_model():
    model = YourModel.from_pretrained('model-name')
    return model
```

### 4. 修改处理函数
```python
def process(data):
    try:
        model = gpu_manager.get_model(load_func=load_model)
        result = model(data)
        gpu_manager.force_offload()  # 关键！
        return result
    except Exception as e:
        gpu_manager.force_offload()  # 异常也要卸载
        raise e
```

## 📡 API 端点

```python
# 查看状态
GET /gpu/status

# 手动卸载
POST /gpu/offload

# 完全释放
POST /gpu/release
```

## 🎨 伪代码模板

```python
# ============ 初始化 ============
gpu_manager = GPUResourceManager(idle_timeout=600)
gpu_manager.start_monitor()

# ============ 处理函数 ============
def process_task(input_data):
    try:
        # 步骤1: 懒加载（自动判断从哪里加载）
        model = gpu_manager.get_model(load_func=load_model)
        
        # 步骤2: 处理
        result = model.predict(input_data)
        
        # 步骤3: 卸载（释放GPU显存到CPU）
        gpu_manager.force_offload()
        
        return result
        
    except Exception as e:
        # 异常处理：确保卸载
        gpu_manager.force_offload()
        raise e

# ============ API 路由 ============
@app.route('/api/process', methods=['POST'])
def api_process():
    data = request.json['data']
    result = process_task(data)
    return jsonify({'result': result})

# ============ 监控端点 ============
@app.route('/gpu/status')
def gpu_status():
    return jsonify(gpu_manager.get_status())
```

## 📈 性能对比

| 场景 | 传统方式 | 使用管理器 | 提升 |
|------|---------|-----------|------|
| 空闲显存 | 3-4 GB | 0.5 GB | **87%↓** |
| 恢复速度 | 20-30s | 2-5s | **5x↑** |
| 并发支持 | 1个服务 | 多个服务 | **N倍** |

## ⚡ 关键点

1. **必须调用 `force_offload()`**：任务完成后立即调用
2. **异常也要卸载**：在 `except` 块中也调用
3. **线程安全**：管理器自动处理并发
4. **监控启动**：记得调用 `start_monitor()`

## 🔍 验证方法

```bash
# 1. 查看 API 状态
curl http://localhost:5000/gpu/status

# 2. 查看实际显存
nvidia-smi --query-gpu=memory.used --format=csv

# 3. 预期结果
# - 空闲时: < 1GB
# - 处理时: 3-4GB
# - 完成后: < 1GB
```

## 📚 完整文档

- `GPU_MEMORY_BEST_PRACTICES.md` - 详细实现指南
- `GPU_MANAGEMENT.md` - 使用文档
- `gpu_manager.py` - 源代码

## 🎯 一句话总结

**用完就卸载，需要时快速恢复，长期不用完全释放。**
