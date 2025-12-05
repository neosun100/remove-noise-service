import os
import time
import threading
import logging
import torch

logger = logging.getLogger(__name__)

class GPUResourceManager:
    """GPU 资源管理器 - 自动释放和重载"""
    
    def __init__(self, idle_timeout=600):
        self.idle_timeout = idle_timeout  # 默认10分钟
        self.last_use_time = time.time()
        self.model = None
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
            time.sleep(60)  # 每分钟检查一次
            
            with self.lock:
                if self.model is not None:
                    idle_time = time.time() - self.last_use_time
                    if idle_time > self.idle_timeout:
                        logger.info(f"GPU 空闲 {idle_time:.0f}秒，释放资源...")
                        self._release_model()
    
    def _release_model(self):
        """释放模型资源"""
        if self.model is not None:
            del self.model
            self.model = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("✅ GPU 资源已释放")
    
    def get_model(self):
        """获取模型（自动加载）"""
        with self.lock:
            self.last_use_time = time.time()
            
            if self.model is None:
                logger.info("🔄 加载降噪模型...")
                from modelscope.pipelines import pipeline
                from modelscope.utils.constant import Tasks
                
                self.model = pipeline(
                    Tasks.acoustic_noise_suppression,
                    model='damo/speech_zipenhancer_ans_multiloss_16k_base'
                )
                logger.info("✅ 模型加载完成")
            
            return self.model
    
    def update_activity(self):
        """更新活动时间"""
        self.last_use_time = time.time()
    
    def get_idle_time(self):
        """获取空闲时间"""
        return time.time() - self.last_use_time
    
    def is_model_loaded(self):
        """检查模型是否已加载"""
        with self.lock:
            return self.model is not None
    
    def stop(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
