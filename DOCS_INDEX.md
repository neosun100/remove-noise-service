# 📚 GPU 显存管理 - 文档索引

## 文档结构

```
📁 remove-noise-service/
├── 📘 QUICK_REFERENCE.md              ⭐ 快速参考（5分钟上手）
├── 📗 GPU_MEMORY_BEST_PRACTICES.md    ⭐ 最佳实践（完整指南）
├── 📙 GPU_MANAGEMENT.md               使用文档
├── 📕 IMPLEMENTATION_SUMMARY.md       实现总结
├── 🧪 test_gpu_management.sh          自动化测试
└── 💻 gpu_manager.py                  源代码
```

## 📖 阅读指南

### 🚀 快速开始（5分钟）

**目标**：快速集成到现有项目

1. 阅读 `QUICK_REFERENCE.md`
2. 复制 `gpu_manager.py` 到项目
3. 按照 3 步集成

### 📚 深入理解（30分钟）

**目标**：理解原理和最佳实践

1. 阅读 `GPU_MEMORY_BEST_PRACTICES.md`
   - 核心理念
   - 架构设计
   - 实现细节
   - 完整代码示例

### 🔧 使用参考

**目标**：日常使用和问题排查

1. `GPU_MANAGEMENT.md` - API 使用文档
2. `IMPLEMENTATION_SUMMARY.md` - 功能总结

### 🧪 测试验证

**目标**：验证功能正常

```bash
./test_gpu_management.sh
```

## 📋 核心概念速查

### 三层管理策略

```
GPU 显存 (3-4GB)  ← 处理任务时
    ↕
CPU 内存 (3-4GB)  ← 空闲时缓存
    ↕
磁盘存储          ← 长期不用
```

### 关键方法

| 方法 | 用途 | 调用时机 |
|------|------|---------|
| `get_model()` | 懒加载 | 任务开始前 |
| `force_offload()` | 卸载到CPU | 任务完成后 |
| `force_release()` | 完全释放 | 长期不用 |
| `get_status()` | 查看状态 | 监控调试 |

### 标准流程

```python
try:
    model = gpu_manager.get_model(load_func=load_model)
    result = model(data)
    gpu_manager.force_offload()  # 必须！
    return result
except Exception as e:
    gpu_manager.force_offload()  # 异常也要！
    raise e
```

## 🎯 使用场景

### ✅ 适用场景

- 多个服务共享 GPU
- 间歇性使用的 AI 服务
- 显存受限的环境
- 需要快速响应的 API

### ❌ 不适用场景

- 持续高频使用（每秒多次）
- 实时流式处理
- 模型加载时间 > 处理时间

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| 空闲显存占用 | < 1 GB |
| 处理时显存 | 3-4 GB |
| 首次加载 | 20-30秒 |
| CPU→GPU 恢复 | 2-5秒 |
| GPU→CPU 卸载 | < 2秒 |
| 完全释放 | < 1秒 |

## 🔍 常见问题

### Q: 如何确认显存已释放？

```bash
# 方法1: API 查询
curl http://localhost:5000/gpu/status

# 方法2: nvidia-smi
nvidia-smi --query-gpu=memory.used --format=csv
```

### Q: 为什么要先转到 CPU 而不是直接释放？

**A**: CPU 缓存可以快速恢复（2-5秒），避免频繁从磁盘加载（20-30秒）。

### Q: 如何调整超时时间？

```python
# 根据使用频率调整
gpu_manager = GPUResourceManager(
    idle_timeout=300   # 5分钟
    # idle_timeout=600   # 10分钟（默认）
    # idle_timeout=1800  # 30分钟
)
```

### Q: 多个服务如何共享？

**A**: 每个服务独立使用管理器，通过即用即卸自然实现共享。

## 🛠️ 集成检查清单

- [ ] 复制 `gpu_manager.py` 到项目
- [ ] 初始化管理器并启动监控
- [ ] 定义模型加载函数
- [ ] 修改处理函数（3步：加载→处理→卸载）
- [ ] 异常处理中也调用卸载
- [ ] 添加监控端点（可选）
- [ ] 运行测试验证
- [ ] 检查显存占用

## 📞 技术支持

### 验证方法

```bash
# 1. 运行自动化测试
./test_gpu_management.sh

# 2. 查看服务状态
curl https://noise.aws.xin/gpu/status

# 3. 监控显存
watch -n 1 nvidia-smi
```

### 日志调试

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## 🎓 学习路径

### 初级（1小时）
1. 阅读 `QUICK_REFERENCE.md`
2. 运行测试脚本
3. 集成到简单项目

### 中级（3小时）
1. 阅读 `GPU_MEMORY_BEST_PRACTICES.md`
2. 理解状态机和架构
3. 自定义管理器参数

### 高级（1天）
1. 研究源码实现
2. 优化批处理策略
3. 集成到生产环境

## 📈 版本历史

| 版本 | 日期 | 更新内容 |
|------|------|---------|
| v1.0 | 2025-12-05 | 初始版本，完整功能 |

## 🌟 核心价值

1. **显存占用降低 87%**（空闲时）
2. **恢复速度提升 5倍**（相比重新加载）
3. **支持多服务共享** GPU
4. **开箱即用**，无需修改模型代码

---

**快速开始**: 阅读 `QUICK_REFERENCE.md` → 复制 `gpu_manager.py` → 3步集成 → 完成！

---

## 🚀 开发新项目

### Docker 部署 Prompt 模板

**文档**: `DOCKER_DEPLOYMENT_PROMPT.md` ⭐

**用途**: 未来所有 GPU Docker 项目的标准化部署模板

**包含内容**:
- ✅ 完整的 Docker 化要求
- ✅ GPU 显存智能管理（懒加载 + 即用即卸）
- ✅ 自动选择最空闲 GPU
- ✅ UI + API 双模式
- ✅ 测试验证清单
- ✅ 性能指标要求

**使用方法**:
1. 复制 `DOCKER_DEPLOYMENT_PROMPT.md`
2. 提供给 AI 助手
3. AI 按照模板实现所有功能
4. 验证测试清单

**关键特性**:
```
懒加载: 未加载 → GPU → CPU → GPU
即用即卸: 加载 → 处理 → 卸载
自动选择: nvidia-smi → 最空闲 GPU
```

---

**更新日期**: 2025-12-05  
**版本**: v1.1
