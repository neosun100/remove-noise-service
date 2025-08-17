HOST = "0.0.0.0"
PORT = 5080

import os
import time
import sys
import socket
import requests
from pathlib import Path
import uuid
import json
from urllib.parse import urlparse
import io
import re
from contextlib import redirect_stdout, redirect_stderr
import threading

ROOT_DIR = Path(os.getcwd()).as_posix()
TMPDIR = f'{ROOT_DIR}/tmp'

# 域名和协议配置
CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN', 'noise.aws.xin')
USE_HTTPS = os.environ.get('USE_HTTPS', 'true').lower() == 'true'

# 环境变量配置
os.environ['MODELSCOPE_CACHE'] = ROOT_DIR + "/models"
os.environ['HF_HOME'] = ROOT_DIR + "/models"
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = 'true'

# 创建必要目录
Path(ROOT_DIR + "/models").mkdir(parents=True, exist_ok=True)
Path(TMPDIR).mkdir(exist_ok=True)

# PATH配置
if sys.platform == 'win32':
    os.environ['PATH'] = f'{ROOT_DIR};{ROOT_DIR}\\ffmpeg;' + os.environ['PATH']
else:
    os.environ['PATH'] = f'{ROOT_DIR}:{ROOT_DIR}/ffmpeg:' + os.environ['PATH']

import torch
from torch.backends import cudnn
import torchaudio
import numpy as np
from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import soundfile as sf
import tempfile
import logging
import traceback
from waitress import serve
import subprocess
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# 全局变量
_model_cache = None
_model_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=4)
task_status = {}
task_lock = threading.Lock()

class RealTimeProgressCapture:
    """实时进度捕获器 - 增强版"""
    def __init__(self, task_id, base_progress=80):
        self.task_id = task_id
        self.base_progress = base_progress
        self.progress_range = 19  # 80%-99%
        self.buffer = ""
        self.last_progress = 0
        self.start_time = time.time()
        self.progress_history = []
        
    def write(self, text):
        """实时捕获和解析输出"""
        if text:
            # 保持原始输出到stderr
            sys.__stderr__.write(text)
            sys.__stderr__.flush()
            
            # 添加到缓冲区
            self.buffer += text
            
            # 实时解析进度
            self._parse_progress_realtime(text)
            
        return len(text)
    
    def flush(self):
        sys.__stderr__.flush()
    
    def _parse_progress_realtime(self, text):
        """实时解析进度信息"""
        # 匹配多种进度格式
        patterns = [
            r'current_idx:\s*(\d+)\s+([\d.]+)%',  # current_idx: 7680000 100.00%
            r'progress:\s*([\d.]+)%',              # progress: 50.5%
            r'(\d+)%\s*complete',                  # 75% complete
            r'Processing:\s*([\d.]+)%'             # Processing: 60.5%
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 2:  # current_idx格式
                        model_progress = float(match[1])
                    else:  # 其他格式
                        model_progress = float(match[0] if isinstance(match, tuple) else match)
                    
                    # 记录进度历史
                    current_time = time.time()
                    self.progress_history.append({
                        'progress': model_progress,
                        'timestamp': current_time
                    })
                    
                    # 保留最近10个进度点
                    if len(self.progress_history) > 10:
                        self.progress_history = self.progress_history[-10:]
                    
                    # 计算UI进度
                    ui_progress = self.base_progress + (model_progress / 100.0) * self.progress_range
                    ui_progress = min(99, max(self.last_progress, ui_progress))
                    
                    if ui_progress > self.last_progress:
                        self.last_progress = ui_progress
                        
                        # 计算处理速度和剩余时间
                        speed_info = self._calculate_speed_and_eta(model_progress, current_time)
                        
                        message = f'模型处理中... {model_progress:.1f}%'
                        if speed_info['eta'] > 0:
                            message += f' (预计剩余 {speed_info["eta"]}秒)'
                        
                        # 更新任务进度
                        detailed_info = {
                            'model_progress': model_progress,
                            'processing_speed': speed_info['speed'],
                            'eta_seconds': speed_info['eta'],
                            'elapsed_time': current_time - self.start_time
                        }
                        
                        update_task_progress(
                            self.task_id, 
                            int(ui_progress), 
                            'processing', 
                            message,
                            detailed_info=detailed_info
                        )
                        
                        logger.info(f"任务 {self.task_id[:8]} 实时进度: {model_progress:.1f}% -> UI: {ui_progress:.1f}% (剩余: {speed_info['eta']}s)")
                        
                except (ValueError, IndexError) as e:
                    logger.debug(f"解析进度失败: {e}, text: {text[:100]}")
                    continue
    
    def _calculate_speed_and_eta(self, current_progress, current_time):
        """计算处理速度和预估剩余时间"""
        if len(self.progress_history) < 2:
            return {'speed': 0, 'eta': 0}
        
        try:
            # 计算最近的进度速度
            recent_history = self.progress_history[-5:]  # 最近5个点
            if len(recent_history) >= 2:
                time_diff = recent_history[-1]['timestamp'] - recent_history[0]['timestamp']
                progress_diff = recent_history[-1]['progress'] - recent_history[0]['progress']
                
                if time_diff > 0 and progress_diff > 0:
                    speed = progress_diff / time_diff  # 百分比/秒
                    remaining_progress = 100 - current_progress
                    eta = int(remaining_progress / speed) if speed > 0 else 0
                    
                    # 限制ETA范围
                    eta = max(0, min(eta, 300))  # 最大5分钟
                    
                    return {
                        'speed': round(speed, 2),
                        'eta': eta
                    }
        except Exception as e:
            logger.debug(f"计算ETA失败: {e}")
        
        return {'speed': 0, 'eta': 0}

class AudioProcessingMonitor:
    """音频处理监控器 - 增强版"""
    
    def __init__(self, task_id):
        self.task_id = task_id
        self.start_time = time.time()
        self.stage_times = {}
        
    def update_stage(self, stage, progress, message):
        """更新处理阶段"""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        # 记录阶段时间
        if stage not in self.stage_times:
            self.stage_times[stage] = current_time
        
        detailed_info = {
            'stage': stage,
            'elapsed_time': elapsed,
            'estimated_total': self._estimate_total_time(progress, elapsed),
            'stage_times': self.stage_times.copy()
        }
        
        update_task_progress(
            self.task_id, 
            progress, 
            'processing', 
            message, 
            detailed_info=detailed_info
        )
        
    def _estimate_total_time(self, progress, elapsed):
        """估算总处理时间"""
        if progress > 20:  # 有足够的进度数据
            estimated = (elapsed / progress) * 100
            return max(elapsed, min(estimated, 600))  # 限制在10分钟内
        return None

def get_server_ip():
    """获取服务器IP地址"""
    try:
        response = requests.get('http://ipinfo.io/ip', timeout=5)
        if response.status_code == 200:
            return response.text.strip()
    except:
        pass
    
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "localhost"

SERVER_IP = get_server_ip()

def get_base_url(request=None):
    """获取基础URL，支持代理和自定义域名"""
    if CUSTOM_DOMAIN:
        protocol = 'https' if USE_HTTPS else 'http'
        return f"{protocol}://{CUSTOM_DOMAIN}"
    
    if request:
        forwarded_proto = request.headers.get('X-Forwarded-Proto')
        forwarded_host = request.headers.get('X-Forwarded-Host')
        
        if forwarded_proto and forwarded_host:
            return f"{forwarded_proto}://{forwarded_host}"
        
        if request.headers.get('X-Forwarded-Ssl') == 'on':
            protocol = 'https'
        elif forwarded_proto:
            protocol = forwarded_proto
        else:
            protocol = 'https' if request.is_secure else 'http'
        
        host = forwarded_host or request.headers.get('Host') or f"{SERVER_IP}:{PORT}"
        return f"{protocol}://{host}"
    
    protocol = 'https' if USE_HTTPS else 'http'
    return f"{protocol}://{SERVER_IP}:{PORT}"

def load_config():
    """加载配置"""
    global CUSTOM_DOMAIN, USE_HTTPS
    
    CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN', CUSTOM_DOMAIN)
    USE_HTTPS = os.environ.get('USE_HTTPS', str(USE_HTTPS).lower()).lower() == 'true'
    
    if CUSTOM_DOMAIN:
        logger.info(f"✅ 使用自定义域名: {CUSTOM_DOMAIN}")
        logger.info(f"🔒 使用HTTPS: {USE_HTTPS}")
    else:
        logger.info(f"📡 使用服务器IP: {SERVER_IP}:{PORT}")

def get_model():
    """获取缓存的模型实例"""
    global _model_cache
    if _model_cache is None:
        with _model_lock:
            if _model_cache is None:
                try:
                    from modelscope.pipelines import pipeline
                    from modelscope.utils.constant import Tasks
                    logger.info("正在初始化降噪模型...")
                    _model_cache = pipeline(
                        Tasks.acoustic_noise_suppression,
                        model='damo/speech_zipenhancer_ans_multiloss_16k_base'
                    )
                    logger.info("模型初始化完成")
                except Exception as e:
                    logger.error(f"模型初始化失败: {e}")
                    raise
    return _model_cache

def update_task_progress(task_id, progress, status, message="", result_url="", detailed_info=None):
    """更新任务进度"""
    with task_lock:
        task_status[task_id] = {
            'progress': progress,
            'status': status,
            'message': message,
            'result_url': result_url,
            'timestamp': time.time(),
            'detailed_info': detailed_info or {}
        }

def get_task_status(task_id):
    """获取任务状态"""
    with task_lock:
        return task_status.get(task_id, {'status': 'not_found'})

def validate_audio(audio_path):
    """验证并修复音频文件"""
    try:
        data, samplerate = sf.read(audio_path)
        
        if len(data) == 0:
            logger.error(f"音频文件为空: {audio_path}")
            return False
            
        if np.isnan(data).any() or np.isinf(data).any():
            logger.warning(f"音频包含无效数据，正在修复: {audio_path}")
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
            sf.write(audio_path, data, samplerate)
            
        return True
    except Exception as e:
        logger.error(f"音频验证失败: {e}")
        return False

def clean_filename(filename):
    """清理文件名"""
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext
    return filename

def save_audio(file_data, original_filename, task_id):
    """保存并转换音频文件"""
    try:
        update_task_progress(task_id, 10, 'processing', '正在保存文件...')
        
        clean_name = clean_filename(original_filename)
        filename = re.sub(r"[\"'#\?\><;,=\+\*~!@\$\%\^&\(\)\{\}\|\[\]\s ]+", "", clean_name)
        
        if not filename:
            filename = f"audio_{int(time.time())}.wav"
        
        original_path = f'{TMPDIR}/{filename}'
        with open(original_path, 'wb') as f:
            f.write(file_data)
        
        logger.info(f"文件保存成功: {original_path}, 大小: {len(file_data)} bytes")
        
        update_task_progress(task_id, 30, 'processing', '正在转换音频格式...')
        
        cover_file = f'{Path(filename).stem}-16kconver.wav'
        convert_result = runffmpeg([
            '-y', '-i', original_path,
            '-ar', '16000', '-ac', '1',
            f'{TMPDIR}/{cover_file}'
        ])
        
        if convert_result != "ok":
            logger.error(f"音频转换失败: {convert_result}")
            raise Exception(f"音频转换失败: {convert_result}")
        
        converted_path = f"{TMPDIR}/{cover_file}"
        
        if not validate_audio(converted_path):
            raise Exception("转换后的音频文件无效")
            
        try:
            os.remove(original_path)
        except:
            pass
            
        update_task_progress(task_id, 50, 'processing', '音频转换完成，准备降噪处理...')
        return converted_path
        
    except Exception as e:
        logger.error(f"保存音频失败: {e}")
        update_task_progress(task_id, 0, 'failed', f'文件处理失败: {str(e)}')
        raise

def remove_noise_with_realtime_progress(audio_path, output_file, task_id, base_url=None):
    """带实时进度的音频降噪处理"""
    monitor = AudioProcessingMonitor(task_id)
    
    try:
        monitor.update_stage('validation', 60, '正在验证音频文件...')
        
        if not validate_audio(audio_path):
            logger.error("输入音频验证失败")
            update_task_progress(task_id, 0, 'failed', '输入音频文件无效')
            return audio_path
        
        monitor.update_stage('model_loading', 70, '正在加载降噪模型...')
        ans = get_model()
        
        monitor.update_stage('processing', 80, '模型处理中，请稍候...')
        
        # 创建实时进度捕获器
        progress_capture = RealTimeProgressCapture(task_id, base_progress=80)
        
        # 创建自定义的stdout/stderr
        class TeeOutput:
            def __init__(self, original, capture):
                self.original = original
                self.capture = capture
            
            def write(self, text):
                # 写入原始输出
                result = self.original.write(text)
                self.original.flush()
                # 同时发送给进度捕获器
                self.capture.write(text)
                return result
            
            def flush(self):
                self.original.flush()
                self.capture.flush()
        
        # 保存原始输出
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            # 设置Tee输出，既保持原始输出又捕获进度
            sys.stdout = TeeOutput(original_stdout, progress_capture)
            sys.stderr = TeeOutput(original_stderr, progress_capture)
            
            logger.info(f"开始模型处理: {audio_path}")
            
            # 执行降噪处理
            result = ans(audio_path, output_path=output_file)
            
        finally:
            # 恢复原始输出
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        # 最终验证
        monitor.update_stage('finalizing', 99, '正在完成处理...')
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if not base_url:
                base_url = get_base_url()
            
            result_url = f'{base_url}/tmp/{Path(output_file).name}'
            file_size = os.path.getsize(output_file)
            processing_time = time.time() - monitor.start_time
            
            detailed_info = {
                'stage': 'completed',
                'file_size': file_size,
                'processing_time': processing_time,
                'final_result': True
            }
            
            update_task_progress(
                task_id, 100, 'completed', 
                '降噪处理完成！', result_url, detailed_info
            )
            
            logger.info(f"降噪处理成功: {output_file}, 大小: {file_size} bytes, 耗时: {processing_time:.1f}秒")
            return output_file
        else:
            update_task_progress(task_id, 0, 'failed', '降噪处理失败')
            return audio_path
            
    except Exception as e:
        logger.error(f"降噪处理异常: {e}")
        update_task_progress(task_id, 0, 'failed', f'降噪处理失败: {str(e)}')
        return audio_path

def process_audio_async(converted_path, task_id, base_url=None):
    """异步音频处理"""
    try:
        noise_removed_path = converted_path.replace('-16kconver.wav', '-remove-noise.wav')
        result_path = remove_noise_with_realtime_progress(converted_path, noise_removed_path, task_id, base_url)
        return result_path
    except Exception as e:
        logger.error(f"异步处理失败: {e}")
        update_task_progress(task_id, 0, 'failed', f'处理失败: {str(e)}')
        return None

def runffmpeg(arg):
    """执行FFmpeg命令"""
    try:
        cmd = ["ffmpeg", "-hide_banner", "-y"] + arg
        
        p = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=0 if sys.platform != 'win32' else subprocess.CREATE_NO_WINDOW
        )
        
        try:
            outs, errs = p.communicate(timeout=300)
            
            if p.returncode == 0:
                return "ok"
            else:
                error_msg = errs.decode('utf-8', errors='ignore')
                logger.error(f"FFmpeg错误: {error_msg}")
                return f"FFmpeg错误: {error_msg}"
                
        except subprocess.TimeoutExpired:
            p.kill()
            return "FFmpeg处理超时"
            
    except Exception as e:
        logger.error(f"FFmpeg执行异常: {e}")
        return f"FFmpeg执行失败: {str(e)}"

# ==================== 路由定义 ====================

@app.route('/tmp/<path:filename>')
def tmp_files(filename):
    """安全的临时文件访问"""
    try:
        safe_path = os.path.join(TMPDIR, os.path.basename(filename))
        if os.path.exists(safe_path) and os.path.commonpath([TMPDIR, safe_path]) == TMPDIR:
            logger.info(f"文件下载请求: {filename}, 文件大小: {os.path.getsize(safe_path)} bytes")
            return send_file(safe_path, as_attachment=True)
        else:
            logger.warning(f"文件不存在或路径不安全: {filename}")
            return jsonify({'code': -1, 'msg': '文件不存在'}), 404
    except Exception as e:
        logger.error(f"文件访问错误: {e}")
        return jsonify({'code': -1, 'msg': '文件访问失败'}), 500

@app.route('/')
def index():
    """主页"""
    base_url = get_base_url(request)
    domain_info = CUSTOM_DOMAIN if CUSTOM_DOMAIN else f"{SERVER_IP}:{PORT}"
    
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音频降噪处理系统</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .container {{
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            padding: 40px;
            max-width: 600px;
            width: 90%;
        }}
        
        h1 {{
            text-align: center;
            color: #333;
            margin-bottom: 30px;
            font-size: 2.5em;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        
        .upload-area {{
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 60px 20px;
            text-align: center;
            margin: 30px 0;
            transition: all 0.3s ease;
            cursor: pointer;
            background: #fafafa;
        }}
        
        .upload-area:hover {{
            border-color: #667eea;
            background: #f0f4ff;
            transform: translateY(-2px);
        }}
        
        .upload-area.dragover {{
            border-color: #667eea;
            background: #e8f2ff;
            transform: scale(1.02);
        }}
        
        .upload-icon {{
            font-size: 4em;
            color: #667eea;
            margin-bottom: 20px;
        }}
        
        .upload-text {{
            font-size: 1.2em;
            color: #666;
            margin-bottom: 10px;
        }}
        
        .upload-hint {{
            color: #999;
            font-size: 0.9em;
        }}
        
        #fileInput {{
            display: none;
        }}
        
        .progress-container {{
            display: none;
            margin: 30px 0;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 35px;
            background: #f0f0f0;
            border-radius: 18px;
            overflow: hidden;
            position: relative;
            box-shadow: inset 0 3px 6px rgba(0,0,0,0.1);
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #ffa726, #ff7043);
            transition: all 0.3s ease;
            border-radius: 18px;
            position: relative;
        }}
        
        .progress-fill.stage-converting {{
            background: linear-gradient(90deg, #ffa726, #ff7043);
        }}
        
        .progress-fill.stage-processing {{
            background: linear-gradient(90deg, #667eea, #764ba2);
            animation: processing-glow 2s ease-in-out infinite alternate;
        }}
        
        .progress-fill.stage-completing {{
            background: linear-gradient(90deg, #4CAF50, #45a049);
        }}
        
        @keyframes processing-glow {{
            from {{ box-shadow: 0 0 10px rgba(102, 126, 234, 0.5); }}
            to {{ box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }}
        }}
        
        .progress-text {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            font-size: 1.1em;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }}
        
        .status-text {{
            text-align: center;
            margin-top: 20px;
            font-size: 1.1em;
            color: #555;
            font-weight: 500;
            min-height: 30px;
        }}
        
        .processing-stats {{
            background: linear-gradient(135deg, #f0f4ff, #e8f2ff);
            padding: 15px;
            border-radius: 12px;
            margin: 15px 0;
            font-size: 0.95em;
            color: #555;
            display: none;
            border-left: 4px solid #667eea;
        }}
        
        .stat-item {{
            display: inline-block;
            margin: 5px 15px 5px 0;
            padding: 5px 10px;
            background: rgba(255,255,255,0.7);
            border-radius: 15px;
            font-size: 0.9em;
        }}
        
        .result-container {{
            display: none;
            margin-top: 30px;
            padding: 25px;
            background: linear-gradient(135deg, #f8f9ff, #e8f2ff);
            border-radius: 15px;
            border: 1px solid #e0e8ff;
        }}
        
        .result-title {{
            color: #333;
            font-size: 1.3em;
            margin-bottom: 20px;
            font-weight: 600;
        }}
        
        .download-btn {{
            background: linear-gradient(45deg, #4CAF50, #45a049);
            color: white;
            padding: 15px 30px;
            text-decoration: none;
            border-radius: 25px;
            display: inline-block;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        }}
        
        .download-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(76, 175, 80, 0.4);
        }}
        
        .error {{
            color: #e74c3c;
            background: #ffeaea;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #e74c3c;
            margin: 15px 0;
        }}
        
        .success {{
            color: #27ae60;
            background: #eafaf1;
            padding: 15px;
            border-radius: 10px;
            border-left: 4px solid #27ae60;
            margin: 15px 0;
            font-weight: 500;
        }}
        
        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #eee;
        }}
        
        .info-item:last-child {{
            border-bottom: none;
        }}
        
        .info-label {{
            font-weight: 600;
            color: #555;
        }}
        
        .info-value {{
            color: #333;
            word-break: break-all;
        }}
        
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 10px;
        }}
        
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        
        .footer {{
            text-align: center;
            margin-top: 30px;
            color: #999;
            font-size: 0.9em;
        }}
        
        .url-info {{
            background: #f8f9fa;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-size: 0.85em;
            color: #666;
        }}
        
        @media (max-width: 600px) {{
            .container {{
                padding: 20px;
                margin: 10px;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .upload-area {{
                padding: 40px 15px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎵 音频降噪处理</h1>
        
        <div class="upload-area" id="uploadArea">
            <div class="upload-icon">📁</div>
            <div class="upload-text">点击选择音频文件或拖拽到此处</div>
            <div class="upload-hint">支持 MP3、WAV、M4A 等格式，最大 50MB</div>
            <input type="file" id="fileInput" accept="audio/*">
        </div>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill">
                    <div class="progress-text" id="progressText">0%</div>
                </div>
            </div>
            <div class="status-text" id="statusText">准备中...</div>
            <div class="processing-stats" id="processingStats"></div>
        </div>
        
        <div class="result-container" id="resultContainer">
            <div class="result-title">✅ 处理完成</div>
            <div id="resultContent"></div>
        </div>
        
        <div class="footer">
            <p>🌐 域名: __DOMAIN_INFO__ | 📁 文件将在1小时后自动清理</p>
            <div class="url-info">
                🔗 当前访问地址: __BASE_URL__
            </div>
        </div>
    </div>

    <script>
        let currentTaskId = null;
        let statusCheckInterval = null;
        let startTime = null;
        let lastProgress = 0;
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const progressContainer = document.getElementById('progressContainer');
        const resultContainer = document.getElementById('resultContainer');
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        const statusText = document.getElementById('statusText');
        const processingStats = document.getElementById('processingStats');

        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });

        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                uploadFile(file);
            }
        });

        // 拖拽功能
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                uploadFile(files[0]);
            }
        });

        function uploadFile(file) {
            if (file.size > 50 * 1024 * 1024) {
                showError('文件大小不能超过 50MB');
                return;
            }

            const formData = new FormData();
            formData.append('audio', file);

            progressContainer.style.display = 'block';
            resultContainer.style.display = 'none';
            uploadArea.style.display = 'none';
            startTime = Date.now();
            lastProgress = 0;
            
            updateProgress(5, '正在上传文件...', 'uploading');

            fetch('/upload_async', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.code === 0) {
                    currentTaskId = data.data.task_id;
                    updateProgress(10, '上传完成，开始处理...', 'uploaded');
                    startStatusCheck();
                } else {
                    showError(data.msg);
                }
            })
            .catch(error => {
                showError('上传失败: ' + error.message);
            });
        }

        function startStatusCheck() {
            statusCheckInterval = setInterval(checkStatus, 500); // 更高频率检查
        }

        function checkStatus() {
            if (!currentTaskId) return;

            fetch(`/status/${{currentTaskId}}`)
            .then(response => response.json())
            .then(data => {
                if (data.code === 0) {
                    const taskData = data.data;
                    
                    // 确保进度只增不减
                    const progress = Math.max(lastProgress, taskData.progress);
                    lastProgress = progress;
                    
                    let stage = 'processing';
                    if (progress >= 95) stage = 'completing';
                    else if (progress >= 80) stage = 'processing';
                    else if (progress >= 30) stage = 'converting';
                    
                    updateProgress(progress, taskData.message, stage);
                    
                    // 显示详细统计信息
                    if (taskData.detailed_info) {
                        updateProcessingStats(taskData.detailed_info);
                    }
                    
                    if (taskData.status === 'completed') {
                        showResult(taskData);
                        clearInterval(statusCheckInterval);
                    } else if (taskData.status === 'failed') {
                        showError(taskData.message);
                        clearInterval(statusCheckInterval);
                    }
                }
            })
            .catch(error => {
                console.error('状态检查失败:', error);
            });
        }

        function updateProgress(percent, message, stage = 'processing') {
            progressFill.style.width = percent + '%';
            progressText.textContent = percent + '%';
            
            // 根据阶段设置不同的样式
            progressFill.className = `progress-fill stage-${{stage}}`;
            
            let statusMessage = message;
            let spinner = '<span class="loading-spinner"></span>';
            
            statusText.innerHTML = `${{spinner}}${{statusMessage}}`;
        }

        function updateProcessingStats(detailedInfo) {
            if (!detailedInfo) return;
            
            let statsHtml = '';
            
            if (detailedInfo.elapsed_time) {
                const elapsed = Math.round(detailedInfo.elapsed_time);
                statsHtml += `<span class="stat-item">⏱️ 已处理: ${{elapsed}}秒</span>`;
            }
            
            if (detailedInfo.eta_seconds && detailedInfo.eta_seconds > 0) {
                statsHtml += `<span class="stat-item">⏳ 预计剩余: ${{detailedInfo.eta_seconds}}秒</span>`;
            }
            
            if (detailedInfo.model_progress) {
                statsHtml += `<span class="stat-item">🔧 模型进度: ${{detailedInfo.model_progress.toFixed(1)}}%</span>`;
            }
            
            if (detailedInfo.processing_speed && detailedInfo.processing_speed > 0) {
                statsHtml += `<span class="stat-item">⚡ 处理速度: ${{detailedInfo.processing_speed.toFixed(1)}}%/s</span>`;
            }
            
            if (detailedInfo.stage) {
                const stageNames = {
                    'validation': '验证阶段',
                    'model_loading': '模型加载',
                    'processing': '模型处理',
                    'finalizing': '完成处理'
                };
                statsHtml += `<span class="stat-item">📊 当前: ${{stageNames[detailedInfo.stage] || detailedInfo.stage}}</span>`;
            }
            
            if (statsHtml) {
                processingStats.innerHTML = statsHtml;
                processingStats.style.display = 'block';
            }
        }

        function showResult(data) {
            progressContainer.style.display = 'none';
            resultContainer.style.display = 'block';
            
            const totalTime = data.detailed_info && data.detailed_info.processing_time ? 
                Math.round(data.detailed_info.processing_time) : 
                Math.round((Date.now() - startTime) / 1000);
            
            const fileSize = data.detailed_info && data.detailed_info.file_size ?
                (data.detailed_info.file_size / 1024 / 1024).toFixed(2) + ' MB' :
                '未知';
            
            const resultContent = document.getElementById('resultContent');
            resultContent.innerHTML = `
                <div class="success">🎉 音频降噪处理成功！</div>
                <div class="info-item">
                    <span class="info-label">任务ID:</span>
                    <span class="info-value">${{data.task_id}}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">处理时间:</span>
                    <span class="info-value">${{totalTime}} 秒</span>
                </div>
                <div class="info-item">
                    <span class="info-label">文件大小:</span>
                    <span class="info-value">${{fileSize}}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">完成时间:</span>
                    <span class="info-value">${{new Date(data.timestamp * 1000).toLocaleString()}}</span>
                </div>
                <div class="info-item">
                    <span class="info-label">下载链接:</span>
                    <span class="info-value">${{data.result_url}}</span>
                </div>
                <div style="text-align: center; margin-top: 25px;">
                    <a href="${{data.result_url}}" class="download-btn" download>
                        🎵 下载处理后的音频
                    </a>
                </div>
                <div style="text-align: center; margin-top: 15px;">
                    <button onclick="resetUpload()" style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer;">
                        🔄 处理新文件
                    </button>
                </div>
            `;
        }

        function showError(message) {
            progressContainer.style.display = 'none';
            resultContainer.style.display = 'block';
            uploadArea.style.display = 'block';
            
            const resultContent = document.getElementById('resultContent');
            resultContent.innerHTML = `
                <div class="error">❌ ${{message}}</div>
                <div style="text-align: center; margin-top: 15px;">
                    <button onclick="resetUpload()" style="background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer;">
                        🔄 重新尝试
                    </button>
                </div>
            `;
        }

        function resetUpload() {
            currentTaskId = null;
            startTime = null;
            lastProgress = 0;
            if (statusCheckInterval) {
                clearInterval(statusCheckInterval);
                statusCheckInterval = null;
            }
            
            progressContainer.style.display = 'none';
            resultContainer.style.display = 'none';
            processingStats.style.display = 'none';
            uploadArea.style.display = 'block';
            fileInput.value = '';
        }
    </script>
</body>
</html>
    """
    html_content = (
        html_template
        .replace('__DOMAIN_INFO__', domain_info)
        .replace('__BASE_URL__', base_url)
        .replace('{{', '{').replace('}}', '}' )
    )
    return Response(html_content, mimetype='text/html')

@app.route('/upload_async', methods=['POST'])
def upload_file_async():
    """异步上传处理"""
    try:
        if 'audio' not in request.files:
            return jsonify({'code': -1, 'msg': '必须选择文件上传'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'code': -1, 'msg': '未选择上传文件'})
        
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        
        if size > 50 * 1024 * 1024:
            return jsonify({'code': -1, 'msg': '文件大小不能超过50MB'})
        
        file_data = file.read()
        original_filename = file.filename
        
        logger.info(f"接收到文件: {original_filename}, 大小: {len(file_data)} bytes")
        
        base_url = get_base_url(request)
        logger.info(f"使用基础URL: {base_url}")
        
        task_id = str(uuid.uuid4())
        
        update_task_progress(task_id, 0, 'processing', '开始处理...')
        
        def process_task():
            try:
                converted_path = save_audio(file_data, original_filename, task_id)
                process_audio_async(converted_path, task_id, base_url)
            except Exception as e:
                logger.error(f"任务处理失败: {e}")
                update_task_progress(task_id, 0, 'failed', f'处理失败: {str(e)}')
        
        executor.submit(process_task)
        
        return jsonify({
            'code': 0,
            'msg': '文件上传成功，正在处理中',
            'data': {
                'task_id': task_id,
                'status_url': f'{base_url}/status/{task_id}',
                'estimated_time': '30-90秒'
            }
        })
        
    except Exception as e:
        logger.error(f"异步上传异常: {e}")
        return jsonify({'code': -1, 'msg': f'上传失败: {str(e)}'})

@app.route('/status/<task_id>')
def check_status(task_id):
    """检查处理状态"""
    try:
        status_info = get_task_status(task_id)
        
        if status_info['status'] == 'not_found':
            return jsonify({'code': -1, 'msg': '任务不存在'})
        
        response_data = {
            'task_id': task_id,
            'status': status_info['status'],
            'progress': status_info.get('progress', 0),
            'message': status_info.get('message', ''),
            'timestamp': status_info['timestamp']
        }
        
        if status_info.get('detailed_info'):
            response_data['detailed_info'] = status_info['detailed_info']
        
        if status_info['status'] == 'completed' and status_info.get('result_url'):
            response_data['result_url'] = status_info['result_url']
        
        return jsonify({'code': 0, 'data': response_data})
        
    except Exception as e:
        logger.error(f"状态查询异常: {e}")
        return jsonify({'code': -1, 'msg': '状态查询失败'})

@app.route('/health')
def health_check():
    """健康检查接口"""
    base_url = get_base_url(request)
    return jsonify({
        'status': 'healthy',
        'server_ip': SERVER_IP,
        'port': PORT,
        'custom_domain': CUSTOM_DOMAIN,
        'use_https': USE_HTTPS,
        'base_url': base_url,
        'model_loaded': _model_cache is not None,
        'active_tasks': len(task_status),
        'timestamp': time.time()
    })

@app.route('/api', methods=['GET', 'POST'])
def denoise():
    """降噪处理接口（向后兼容）"""
    try:
        data = {
            "url": request.form.get('url') or request.args.get('url'),
            "stream": int(request.form.get('stream', 0) or request.args.get('stream', 0))
        }
        
        processed_url = data.get('url')
        base_url = get_base_url(request)
        
        if not processed_url:
            if 'audio' not in request.files:
                return jsonify({'code': -1, 'msg': '必须上传文件或传递音频路径'})
            
            file = request.files['audio']
            if file.filename == '':
                return jsonify({'code': -1, 'msg': '未选择上传文件'})
            
            temp_task_id = str(uuid.uuid4())
            file_data = file.read()
            processed_url = save_audio(file_data, file.filename, temp_task_id)
        
        if not os.path.exists(processed_url):
            return jsonify({'code': -1, 'msg': '音频文件不存在'})
        
        out_path = processed_url.replace('-16kconver.wav', '-remove-noise.wav')
        temp_task_id = str(uuid.uuid4())
        result_path = remove_noise_with_realtime_progress(processed_url, out_path, temp_task_id, base_url)
        
        if result_path == processed_url:
            return jsonify({'code': -1, 'msg': '降噪处理失败'})
        
        if data.get('stream') == 1:
            return send_file(
                out_path, 
                mimetype="audio/wav",
                as_attachment=True, 
                download_name='remove-noise.wav'
            )
        
        result_url = f'{base_url}/tmp/{Path(out_path).name}'
        return jsonify({
            'code': 0, 
            'data': {
                'url': result_url
            }
        })
        
    except Exception as e:
        logger.error(f"降噪处理异常: {e}")
        return jsonify({'code': -1, 'msg': f'处理失败: {str(e)}'})

def cleanup_old_files():
    """清理旧文件"""
    try:
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(TMPDIR):
            file_path = os.path.join(TMPDIR, filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getctime(file_path) > 3600:
                    os.remove(file_path)
                    cleaned_count += 1
                    logger.info(f"清理旧文件: {filename}")
        
        with task_lock:
            expired_tasks = [
                task_id for task_id, info in task_status.items()
                if current_time - info['timestamp'] > 7200
            ]
            for task_id in expired_tasks:
                del task_status[task_id]
        
        if cleaned_count > 0 or expired_tasks:
            logger.info(f"清理完成: 文件 {cleaned_count} 个, 任务状态 {len(expired_tasks)} 个")
                
    except Exception as e:
        logger.error(f"清理文件失败: {e}")

def openweb(web_address):
    """打开网页"""
    import webbrowser
    try:
        time.sleep(3)
        webbrowser.open(web_address)
        logger.info(f"打开网页: {web_address}")
    except Exception as e:
        logger.error(f"打开网页失败: {e}")

if __name__ == '__main__':
    try:
        load_config()
        cleanup_old_files()
        
        def periodic_cleanup():
            while True:
                time.sleep(3600)
                cleanup_old_files()
        
        threading.Thread(target=periodic_cleanup, daemon=True).start()
        
        print(f"🚀 音频降噪服务启动成功！")
        print(f"📱 本地访问: http://127.0.0.1:{PORT}")
        
        if CUSTOM_DOMAIN:
            protocol = 'https' if USE_HTTPS else 'http'
            print(f"🌐 公网访问: {protocol}://{CUSTOM_DOMAIN}")
            print(f"💊 健康检查: {protocol}://{CUSTOM_DOMAIN}/health")
        else:
            print(f"🌐 公网访问: http://{SERVER_IP}:{PORT}")
            print(f"💊 健康检查: http://{SERVER_IP}:{PORT}/health")
        
        print(f"📊 服务器IP: {SERVER_IP}")
        print(f"🔧 自定义域名: {CUSTOM_DOMAIN or '未配置'}")
        print(f"🔒 HTTPS模式: {'启用' if USE_HTTPS else '禁用'}")
        
        threading.Thread(target=openweb, args=(f'http://127.0.0.1:{PORT}',)).start()
        
        serve(app, host=HOST, port=PORT, threads=8)
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        logger.error(traceback.format_exc())
