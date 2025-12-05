import os
import time
import threading
import logging
import torch
import gc

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """GPU 资源管理器 - 支持 CPU/GPU 转移和完全卸载"""
    
    def __init__(self, idle_timeout=600):
        self.idle_timeout = idle_timeout
        self.last_use_time = time.time()
        self.model = None
        self.model_on_cpu = None  # CPU 缓存的模型
        self.device = None
        self.lock = threading.Lock()
        self.monitor_thread = None
        self.running = False
        
    def start_monitor(self):
        """启动监控线程"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info(f"GPU 资源监控已启动，空闲超时: {self.idle_timeout}秒")
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            time.sleep(30)  # 每30秒检查一次
            
            with self.lock:
                if self.model is not None:
                    idle_time = time.time() - self.last_use_time
                    if idle_time > self.idle_timeout:
                        logger.info(f"GPU 空闲 {idle_time:.0f}秒，转移到 CPU...")
                        self._move_to_cpu()
    
    def _move_to_cpu(self):
        """将模型从 GPU 转移到 CPU"""
        if self.model is None:
            return
            
        try:
            # 保存模型到 CPU
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'cpu'):
                logger.info("🔄 转移模型到 CPU...")
                self.model.model.cpu()
                self.model_on_cpu = self.model
                self.model = None
                
                # 清理 GPU 缓存
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    gc.collect()
                
                logger.info("✅ 模型已转移到 CPU，GPU 显存已释放")
            else:
                # 如果无法转移，直接释放
                self._release_all()
        except Exception as e:
            logger.error(f"转移模型到 CPU 失败: {e}")
            self._release_all()
    
    def _move_to_gpu(self):
        """将模型从 CPU 转移到 GPU"""
        if self.model_on_cpu is None:
            return False
            
        try:
            logger.info("🔄 转移模型到 GPU...")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            if hasattr(self.model_on_cpu, 'model') and hasattr(self.model_on_cpu.model, 'to'):
                self.model_on_cpu.model.to(device)
                self.model = self.model_on_cpu
                self.model_on_cpu = None
                self.device = device
                
                logger.info(f"✅ 模型已转移到 {device}")
                return True
        except Exception as e:
            logger.error(f"转移模型到 GPU 失败: {e}")
            return False
        
        return False
    
    def _release_all(self):
        """完全释放所有模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.model_on_cpu is not None:
            del self.model_on_cpu
            self.model_on_cpu = None
        
        self.device = None
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("✅ 所有模型资源已释放")
    
    def get_model(self):
        """获取模型（自动加载或从 CPU 转移）"""
        with self.lock:
            self.last_use_time = time.time()
            
            # 如果模型在 GPU 上，直接返回
            if self.model is not None:
                return self.model
            
            # 如果模型在 CPU 上，转移到 GPU
            if self.model_on_cpu is not None:
                if self._move_to_gpu():
                    return self.model
                else:
                    # 转移失败，重新加载
                    self._release_all()
            
            # 加载新模型
            logger.info("🔄 加载降噪模型...")
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks
            
            # 不传递 device 参数，让 pipeline 自动选择
            self.model = pipeline(
                Tasks.acoustic_noise_suppression,
                model='damo/speech_zipenhancer_ans_multiloss_16k_base'
            )
            
            # 记录实际使用的设备
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logger.info(f"✅ 模型加载完成 (设备: cuda)")
            else:
                self.device = torch.device('cpu')
                logger.info(f"✅ 模型加载完成 (设备: cpu)")
            
            return self.model
    
    def force_offload(self):
        """强制卸载到 CPU"""
        with self.lock:
            if self.model is not None:
                logger.info("🔄 强制卸载 GPU 显存...")
                self._move_to_cpu()
                return True
            return False
    
    def force_release(self):
        """强制完全释放"""
        with self.lock:
            logger.info("🔄 强制释放所有资源...")
            self._release_all()
            return True
    
    def update_activity(self):
        """更新活动时间"""
        self.last_use_time = time.time()
    
    def get_idle_time(self):
        """获取空闲时间"""
        return time.time() - self.last_use_time
    
    def is_model_loaded(self):
        """检查模型是否在 GPU 上"""
        with self.lock:
            return self.model is not None
    
    def is_model_cached(self):
        """检查模型是否在 CPU 缓存中"""
        with self.lock:
            return self.model_on_cpu is not None
    
    def get_status(self):
        """获取详细状态"""
        with self.lock:
            return {
                'model_on_gpu': self.model is not None,
                'model_on_cpu': self.model_on_cpu is not None,
                'device': str(self.device) if self.device else None,
                'idle_time': int(self.get_idle_time()),
                'idle_timeout': self.idle_timeout
            }
    
    def stop(self):
        """停止监控并释放资源"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self._release_all()
