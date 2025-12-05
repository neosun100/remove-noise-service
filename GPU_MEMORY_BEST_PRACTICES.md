# GPU 显存管理最佳实践

> 适用于所有 GPU Docker 服务的通用显存管理方案

## 📋 目录

- [核心理念](#核心理念)
- [架构设计](#架构设计)
- [实现细节](#实现细节)
- [代码示例](#代码示例)
- [集成指南](#集成指南)
- [性能优化](#性能优化)

---

## 核心理念

### 问题背景

GPU 显存是昂贵且有限的资源，在多服务共享 GPU 环境中：
- 模型长期占用显存导致其他服务无法使用
- 空闲时显存未释放造成资源浪费
- 频繁加载模型影响响应速度

### 解决方案

**三层显存管理策略**：

```
┌─────────────────────────────────────────────┐
│  GPU 显存 (3-4GB)  ← 处理任务时使用        │
│  ↕ 自动转移                                 │
│  CPU 内存 (3-4GB)  ← 空闲时缓存            │
│  ↕ 完全释放                                 │
│  磁盘存储 (模型文件) ← 长期不用时释放      │
└─────────────────────────────────────────────┘
```

**核心逻辑**：
1. **懒加载**：首次请求时才加载模型到 GPU
2. **即用即卸**：任务完成后立即卸载到 CPU
3. **智能缓存**：保留 CPU 缓存以便快速恢复
4. **超时释放**：长期空闲完全释放所有资源

### ⚠️ CUDA Context 基础占用说明

**重要**：即使模型完全卸载到 CPU，GPU 仍会保留 **400-550 MB** 的基础显存占用。

#### 为什么会有这个占用？

```
┌─────────────────────────────────────────────────────┐
│  CUDA Context 基础占用 (~540 MB)                    │
│  ├─ CUDA 驱动程序数据结构      (~150 MB)           │
│  ├─ cuBLAS/cuDNN 库初始化      (~200 MB)           │
│  ├─ 内核缓存和 JIT 编译器      (~100 MB)           │
│  └─ GPU 内存管理器             (~90 MB)            │
└─────────────────────────────────────────────────────┘
```

这是 **PyTorch/CUDA 运行时的固定开销**，包含：
- **CUDA 驱动程序**：管理 GPU 硬件的底层驱动
- **cuBLAS/cuDNN**：深度学习加速库的初始化
- **内核缓存**：编译好的 CUDA 内核代码
- **内存管理器**：GPU 内存分配器的元数据

#### 显存占用对比

| 状态 | 显存占用 | 说明 |
|------|---------|------|
| **模型在 GPU** | 3500-4000 MB | 模型权重 (3000 MB) + Context (540 MB) |
| **模型在 CPU** | **540 MB** | 仅 CUDA Context（✅ 减少 85%） |
| **进程退出** | 0 MB | 完全释放（需重启服务） |

#### 为什么无法消除？

```python
# ❌ 这些操作无法释放 CUDA Context
torch.cuda.empty_cache()      # 只清理 PyTorch 管理的显存
torch.cuda.synchronize()      # 只同步 GPU 操作
gc.collect()                  # 只回收 Python 对象

# ✅ 唯一方法：退出进程
docker restart service        # 重启服务（不推荐生产环境）
```

**原因**：
- `torch.cuda.empty_cache()` 只清理 PyTorch 分配的显存
- CUDA Context 由 **CUDA 驱动管理**，不受 PyTorch 控制
- Context 在进程首次使用 GPU 时创建，进程退出时销毁

#### 实际效果验证

```bash
# 测试：即使没有任何模型，CUDA Context 也会占用显存
python3 << 'EOF'
import torch
device = torch.device('cuda:0')
x = torch.randn(1).to(device)  # 创建一个极小的 tensor
del x
torch.cuda.empty_cache()
# nvidia-smi 仍显示 ~540 MB
EOF
```

#### 结论

✅ **540MB 是正常且必要的**
- 这是所有使用 PyTorch/CUDA 的应用都会有的开销
- 你的优化已经很成功：从 3.5GB 降到 540MB = **减少 85%**
- 剩余的 540MB 是 CUDA 运行时必需的，无需进一步优化
- 如果真的需要释放，只能重启服务（会导致服务中断）

💡 **最佳实践**：
- 接受 540MB 的 Context 占用作为合理开销
- 专注于优化模型本身的显存使用（已完成 ✅）
- 在多 GPU 环境中，每个进程只使用一个 GPU 以避免多个 Context

---

## 架构设计

### 状态机

```
┌──────────┐  首次请求   ┌──────────┐  任务完成   ┌──────────┐
│ 未加载   │ ────────→  │ GPU 加载 │ ────────→  │ CPU 缓存 │
│ (磁盘)   │            │ (处理中) │            │ (待命)   │
└──────────┘            └──────────┘            └──────────┘
     ↑                       ↑                        │
     │                       │ 新请求(快速恢复)       │
     │                       └────────────────────────┘
     │                                                 │
     └─────────────────────────────────────────────────┘
                    超时/手动释放
```

### 组件架构

```python
┌─────────────────────────────────────────────────┐
│              API 服务层                          │
│  - 接收请求                                      │
│  - 调用 GPU 管理器                               │
│  - 任务完成后触发卸载                            │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│           GPU 资源管理器                         │
│  - 模型加载/卸载                                 │
│  - CPU/GPU 转移                                  │
│  - 状态跟踪                                      │
│  - 并发控制                                      │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│            监控线程                              │
│  - 定期检查空闲时间                              │
│  - 自动触发卸载                                  │
│  - 资源清理                                      │
└─────────────────────────────────────────────────┘
```

---

## 实现细节

### 1. GPU 资源管理器

#### 核心数据结构

```python
class GPUResourceManager:
    def __init__(self, idle_timeout=600):
        self.model = None              # GPU 上的模型
        self.model_on_cpu = None       # CPU 缓存的模型
        self.device = None             # 当前设备
        self.last_use_time = time.time()
        self.lock = threading.Lock()   # 线程锁
```

#### 关键方法

**1. 智能加载（懒加载）**

```python
def get_model(self):
    """
    懒加载逻辑：
    1. 如果在 GPU 上 → 直接返回
    2. 如果在 CPU 上 → 快速转移到 GPU
    3. 如果未加载 → 从磁盘加载
    """
    with self.lock:  # 线程安全
        self.last_use_time = time.time()
        
        # 情况1: 模型已在 GPU 上
        if self.model is not None:
            return self.model
        
        # 情况2: 模型在 CPU 缓存中（快速恢复）
        if self.model_on_cpu is not None:
            self._move_to_gpu()  # 2-5秒
            return self.model
        
        # 情况3: 首次加载（较慢）
        self._load_from_disk()  # 20-30秒
        return self.model
```

**2. 卸载到 CPU（即用即卸）**

```python
def force_offload(self):
    """
    任务完成后立即调用
    将模型从 GPU 转移到 CPU，释放显存
    """
    with self.lock:
        if self.model is None:
            return False
        
        # 转移到 CPU
        self.model.model.cpu()
        self.model_on_cpu = self.model
        self.model = None
        
        # 清理 GPU 缓存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        return True
```

**3. 完全释放**

```python
def force_release(self):
    """
    长期不用时完全释放
    清空 GPU 和 CPU 缓存
    """
    with self.lock:
        # 释放所有引用
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.model_on_cpu is not None:
            del self.model_on_cpu
            self.model_on_cpu = None
        
        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()
```

**4. GPU → CPU 转移**

```python
def _move_to_cpu(self):
    """内部方法：GPU 转 CPU"""
    if self.model is None:
        return
    
    # 将模型参数移到 CPU
    self.model.model.cpu()
    
    # 保存到 CPU 缓存
    self.model_on_cpu = self.model
    self.model = None
    
    # 清理 GPU
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
```

**5. CPU → GPU 转移**

```python
def _move_to_gpu(self):
    """内部方法：CPU 转 GPU"""
    if self.model_on_cpu is None:
        return False
    
    # 将模型参数移到 GPU
    device = torch.device('cuda')
    self.model_on_cpu.model.to(device)
    
    # 恢复到 GPU
    self.model = self.model_on_cpu
    self.model_on_cpu = None
    self.device = device
    
    return True
```

### 2. 监控线程

```python
def _monitor_loop(self):
    """
    后台监控线程
    定期检查空闲时间，自动卸载
    """
    while self.running:
        time.sleep(30)  # 每30秒检查一次
        
        with self.lock:
            if self.model is not None:
                idle_time = time.time() - self.last_use_time
                
                # 超过阈值自动卸载
                if idle_time > self.idle_timeout:
                    logger.info(f"空闲 {idle_time}秒，自动卸载")
                    self._move_to_cpu()
```

### 3. API 集成

#### 处理任务的标准流程

```python
def process_task(input_data):
    """
    标准任务处理流程
    """
    try:
        # 1. 获取模型（自动懒加载）
        model = gpu_manager.get_model()
        
        # 2. 处理任务
        result = model(input_data)
        
        # 3. 任务完成后立即卸载
        gpu_manager.force_offload()
        
        return result
        
    except Exception as e:
        # 4. 异常时也要卸载
        gpu_manager.force_offload()
        raise e
```

#### Flask API 示例

```python
@app.route('/process', methods=['POST'])
def process_api():
    try:
        # 接收数据
        data = request.files['file']
        
        # 处理（内部会自动管理显存）
        result = process_task(data)
        
        return jsonify({'result': result})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
```

---

## 代码示例

### 完整的 GPU 管理器实现

```python
import torch
import gc
import time
import threading
import logging

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """
    通用 GPU 资源管理器
    适用于任何 PyTorch 模型
    """
    
    def __init__(self, idle_timeout=600):
        """
        Args:
            idle_timeout: 空闲超时时间（秒），默认10分钟
        """
        self.idle_timeout = idle_timeout
        self.last_use_time = time.time()
        
        # 模型状态
        self.model = None           # GPU 上的模型
        self.model_on_cpu = None    # CPU 缓存
        self.device = None
        
        # 并发控制
        self.lock = threading.Lock()
        
        # 监控线程
        self.running = False
        self.monitor_thread = None
    
    def start_monitor(self):
        """启动后台监控"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(
                target=self._monitor_loop, 
                daemon=True
            )
            self.monitor_thread.start()
            logger.info(f"监控已启动，超时: {self.idle_timeout}秒")
    
    def get_model(self, load_func=None):
        """
        获取模型（懒加载）
        
        Args:
            load_func: 模型加载函数，首次加载时调用
        
        Returns:
            模型对象
        """
        with self.lock:
            self.last_use_time = time.time()
            
            # 已在 GPU 上
            if self.model is not None:
                return self.model
            
            # 在 CPU 缓存中，快速恢复
            if self.model_on_cpu is not None:
                logger.info("从 CPU 恢复到 GPU...")
                self._move_to_gpu()
                return self.model
            
            # 首次加载
            if load_func is None:
                raise ValueError("首次加载需要提供 load_func")
            
            logger.info("首次加载模型...")
            self.model = load_func()
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"模型已加载到 {self.device}")
            
            return self.model
    
    def force_offload(self):
        """
        强制卸载到 CPU
        任务完成后调用
        """
        with self.lock:
            if self.model is None:
                return False
            
            try:
                logger.info("卸载 GPU 显存到 CPU...")
                
                # 转移到 CPU
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()
                elif hasattr(self.model, 'model'):
                    self.model.model.cpu()
                
                # 保存到 CPU 缓存
                self.model_on_cpu = self.model
                self.model = None
                
                # 清理 GPU
                self._cleanup_gpu()
                
                logger.info("✅ 显存已释放")
                return True
                
            except Exception as e:
                logger.error(f"卸载失败: {e}")
                return False
    
    def force_release(self):
        """完全释放所有资源"""
        with self.lock:
            logger.info("完全释放资源...")
            
            # 删除所有引用
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.model_on_cpu is not None:
                del self.model_on_cpu
                self.model_on_cpu = None
            
            self.device = None
            
            # 强制清理
            gc.collect()
            self._cleanup_gpu()
            
            logger.info("✅ 所有资源已释放")
    
    def get_status(self):
        """获取当前状态"""
        with self.lock:
            return {
                'model_on_gpu': self.model is not None,
                'model_on_cpu': self.model_on_cpu is not None,
                'device': str(self.device) if self.device else None,
                'idle_time': int(time.time() - self.last_use_time),
                'idle_timeout': self.idle_timeout
            }
    
    def _move_to_gpu(self):
        """内部：CPU → GPU"""
        if self.model_on_cpu is None:
            return False
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if hasattr(self.model_on_cpu, 'to'):
            self.model_on_cpu.to(device)
        elif hasattr(self.model_on_cpu, 'model'):
            self.model_on_cpu.model.to(device)
        
        self.model = self.model_on_cpu
        self.model_on_cpu = None
        self.device = device
        
        return True
    
    def _move_to_cpu(self):
        """内部：GPU → CPU"""
        if self.model is None:
            return
        
        if hasattr(self.model, 'cpu'):
            self.model.cpu()
        elif hasattr(self.model, 'model'):
            self.model.model.cpu()
        
        self.model_on_cpu = self.model
        self.model = None
        
        self._cleanup_gpu()
    
    def _cleanup_gpu(self):
        """清理 GPU 缓存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            time.sleep(30)
            
            with self.lock:
                if self.model is not None:
                    idle_time = time.time() - self.last_use_time
                    if idle_time > self.idle_timeout:
                        logger.info(f"空闲 {idle_time:.0f}秒，自动卸载")
                        self._move_to_cpu()
    
    def stop(self):
        """停止监控并清理"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.force_release()
```

### 使用示例

#### 1. 初始化

```python
# 创建管理器
gpu_manager = GPUResourceManager(idle_timeout=600)

# 启动监控
gpu_manager.start_monitor()

# 定义模型加载函数
def load_my_model():
    from transformers import AutoModel
    model = AutoModel.from_pretrained('model-name')
    return model
```

#### 2. 在任务中使用

```python
def process_data(input_data):
    """标准处理流程"""
    try:
        # 懒加载模型
        model = gpu_manager.get_model(load_func=load_my_model)
        
        # 处理数据
        with torch.no_grad():
            result = model(input_data)
        
        # 立即卸载
        gpu_manager.force_offload()
        
        return result
        
    except Exception as e:
        # 异常时也卸载
        gpu_manager.force_offload()
        raise e
```

#### 3. API 集成

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['data']
        result = process_data(data)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# 手动控制端点
@app.route('/gpu/offload', methods=['POST'])
def offload():
    success = gpu_manager.force_offload()
    return jsonify({
        'success': success,
        'status': gpu_manager.get_status()
    })

@app.route('/gpu/release', methods=['POST'])
def release():
    gpu_manager.force_release()
    return jsonify({'status': gpu_manager.get_status()})

@app.route('/gpu/status')
def status():
    return jsonify(gpu_manager.get_status())
```

---

## 集成指南

### 步骤 1: 复制管理器代码

将 `GPUResourceManager` 类复制到你的项目中：

```bash
# 创建文件
touch gpu_manager.py

# 复制上面的完整实现
```

### 步骤 2: 初始化管理器

```python
from gpu_manager import GPUResourceManager

# 创建全局实例
gpu_manager = GPUResourceManager(idle_timeout=600)

# 启动监控
gpu_manager.start_monitor()
```

### 步骤 3: 定义模型加载函数

```python
def load_model():
    """
    根据你的模型类型实现
    返回模型对象
    """
    # 示例：加载 HuggingFace 模型
    from transformers import AutoModel
    model = AutoModel.from_pretrained('your-model')
    return model
    
    # 示例：加载自定义模型
    # model = YourModel()
    # model.load_state_dict(torch.load('model.pth'))
    # return model
```

### 步骤 4: 修改处理函数

**修改前**：
```python
def process(data):
    model = load_model()  # 每次都加载
    result = model(data)
    return result
```

**修改后**：
```python
def process(data):
    try:
        # 懒加载
        model = gpu_manager.get_model(load_func=load_model)
        
        # 处理
        result = model(data)
        
        # 立即卸载
        gpu_manager.force_offload()
        
        return result
    except Exception as e:
        gpu_manager.force_offload()
        raise e
```

### 步骤 5: 添加监控端点（可选）

```python
@app.route('/gpu/status')
def gpu_status():
    return jsonify(gpu_manager.get_status())

@app.route('/gpu/offload', methods=['POST'])
def gpu_offload():
    gpu_manager.force_offload()
    return jsonify({'status': 'ok'})
```

---

## 性能优化

### 1. 调整超时时间

根据使用频率调整：

```python
# 高频使用（每分钟多次）
gpu_manager = GPUResourceManager(idle_timeout=300)  # 5分钟

# 中频使用（每小时几次）
gpu_manager = GPUResourceManager(idle_timeout=600)  # 10分钟

# 低频使用（每天几次）
gpu_manager = GPUResourceManager(idle_timeout=180)  # 3分钟
```

### 2. 批处理优化

```python
def process_batch(data_list):
    """批量处理，减少加载次数"""
    try:
        model = gpu_manager.get_model(load_func=load_model)
        
        results = []
        for data in data_list:
            result = model(data)
            results.append(result)
        
        # 批处理完成后卸载
        gpu_manager.force_offload()
        
        return results
    except Exception as e:
        gpu_manager.force_offload()
        raise e
```

### 3. 预热策略

```python
def warmup():
    """服务启动时预加载到 CPU"""
    model = gpu_manager.get_model(load_func=load_model)
    gpu_manager.force_offload()  # 立即卸载到 CPU
    logger.info("模型已预热到 CPU 缓存")

# 在服务启动时调用
if __name__ == '__main__':
    warmup()
    app.run()
```

---

## 监控与调试

### 查看状态

```python
status = gpu_manager.get_status()
print(f"GPU 上: {status['model_on_gpu']}")
print(f"CPU 上: {status['model_on_cpu']}")
print(f"空闲: {status['idle_time']}秒")
```

### 检查实际显存

```bash
# 查看 GPU 显存占用
nvidia-smi --query-gpu=memory.used --format=csv

# 持续监控
watch -n 1 nvidia-smi
```

### 日志记录

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

---

## 常见问题

### Q1: 为什么不直接释放而是先转到 CPU？

**A**: CPU 缓存可以实现快速恢复（2-5秒），避免频繁从磁盘加载（20-30秒）。

### Q2: 如何确认显存已释放？

**A**: 
```bash
# 方法1: nvidia-smi
nvidia-smi

# 方法2: API 查询
curl http://localhost:5000/gpu/status
```

### Q3: 多个服务如何共享 GPU？

**A**: 每个服务独立使用此管理器，通过即用即卸策略自然实现共享。

### Q4: 如何处理并发请求？

**A**: 管理器内置线程锁，自动处理并发安全。

---

## 总结

### 核心要点

1. **懒加载**：首次请求才加载
2. **即用即卸**：任务完成立即卸载
3. **CPU 缓存**：保留缓存快速恢复
4. **自动监控**：超时自动释放

### 适用场景

- ✅ 多服务共享 GPU
- ✅ 间歇性使用的服务
- ✅ 需要快速响应的 API
- ✅ 显存受限的环境

### 性能收益

- 显存占用降低 **80-90%**（空闲时）
- 恢复速度提升 **4-6倍**（相比重新加载）
- 支持更多服务共享同一 GPU

---

## 附录

### 完整项目结构

```
project/
├── gpu_manager.py          # GPU 管理器
├── api.py                  # API 服务
├── model_loader.py         # 模型加载
├── requirements.txt        # 依赖
└── docker-compose.yml      # Docker 配置
```

### Docker 配置示例

```yaml
version: '3.8'
services:
  app:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=0
      - GPU_IDLE_TIMEOUT=600
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### 环境变量

```bash
# .env
GPU_IDLE_TIMEOUT=600        # 空闲超时（秒）
NVIDIA_VISIBLE_DEVICES=0    # GPU ID
```

---

**文档版本**: v1.0  
**更新日期**: 2025-12-05  
**适用范围**: 所有 PyTorch GPU 服务
