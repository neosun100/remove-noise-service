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
CUSTOM_DOMAIN = os.environ.get('CUSTOM_DOMAIN', '')
USE_HTTPS = os.environ.get('USE_HTTPS', 'true').lower() == 'true'
GPU_IDLE_TIMEOUT = int(os.environ.get('GPU_IDLE_TIMEOUT', '10')) * 60  # 转换为秒

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
from flasgger import Swagger, swag_from
import soundfile as sf
import tempfile
import logging
import traceback
from waitress import serve
import subprocess
from functools import lru_cache
import hashlib
from concurrent.futures import ThreadPoolExecutor

from gpu_manager import GPUResourceManager

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Swagger 配置
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": 'apispec',
            "route": '/apispec.json',
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True,
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs"
}

swagger_template = {
    "swagger": "2.0",
    "info": {
        "title": "音频降噪 API",
        "description": "基于 ZipEnhancer 的音频降噪处理服务",
        "version": "2.0.0",
        "contact": {
            "name": "API Support",
            "url": "https://github.com/yourusername/remove-noise-service"
        }
    },
    "host": CUSTOM_DOMAIN or f"{socket.gethostbyname(socket.gethostname())}:{PORT}",
    "basePath": "/",
    "schemes": ["https" if USE_HTTPS else "http"],
    "consumes": ["multipart/form-data", "application/json"],
    "produces": ["application/json", "audio/wav"]
}

swagger = Swagger(app, config=swagger_config, template=swagger_template)

# GPU 资源管理器
gpu_manager = GPUResourceManager(idle_timeout=GPU_IDLE_TIMEOUT)
gpu_manager.start_monitor()

# 全局变量
executor = ThreadPoolExecutor(max_workers=4)
task_status = {}
task_lock = threading.Lock()

# 从原 api.py 导入核心类
class RealTimeProgressCapture:
    """实时进度捕获器"""
    def __init__(self, task_id, base_progress=80):
        self.task_id = task_id
        self.base_progress = base_progress
        self.progress_range = 19
        self.buffer = ""
        self.last_progress = 0
        self.start_time = time.time()
        self.progress_history = []
        
    def write(self, text):
        if text:
            sys.__stderr__.write(text)
            sys.__stderr__.flush()
            self.buffer += text
            self._parse_progress_realtime(text)
        return len(text)
    
    def flush(self):
        sys.__stderr__.flush()
    
    def _parse_progress_realtime(self, text):
        patterns = [
            r'current_idx:\s*(\d+)\s+([\d.]+)%',
            r'progress:\s*([\d.]+)%',
            r'(\d+)%\s*complete',
            r'Processing:\s*([\d.]+)%'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                try:
                    if len(match) == 2:
                        model_progress = float(match[1])
                    else:
                        model_progress = float(match[0] if isinstance(match, tuple) else match)
                    
                    current_time = time.time()
                    self.progress_history.append({
                        'progress': model_progress,
                        'timestamp': current_time
                    })
                    
                    if len(self.progress_history) > 10:
                        self.progress_history = self.progress_history[-10:]
                    
                    ui_progress = self.base_progress + (model_progress / 100.0) * self.progress_range
                    ui_progress = min(99, max(self.last_progress, ui_progress))
                    
                    if ui_progress > self.last_progress:
                        self.last_progress = ui_progress
                        speed_info = self._calculate_speed_and_eta(model_progress, current_time)
                        
                        message = f'模型处理中... {model_progress:.1f}%'
                        if speed_info['eta'] > 0:
                            message += f' (预计剩余 {speed_info["eta"]}秒)'
                        
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
                        
                except (ValueError, IndexError):
                    continue
    
    def _calculate_speed_and_eta(self, current_progress, current_time):
        if len(self.progress_history) < 2:
            return {'speed': 0, 'eta': 0}
        
        try:
            recent_history = self.progress_history[-5:]
            if len(recent_history) >= 2:
                time_diff = recent_history[-1]['timestamp'] - recent_history[0]['timestamp']
                progress_diff = recent_history[-1]['progress'] - recent_history[0]['progress']
                
                if time_diff > 0 and progress_diff > 0:
                    speed = progress_diff / time_diff
                    remaining_progress = 100 - current_progress
                    eta = int(remaining_progress / speed) if speed > 0 else 0
                    eta = max(0, min(eta, 300))
                    
                    return {'speed': round(speed, 2), 'eta': eta}
        except Exception:
            pass
        
        return {'speed': 0, 'eta': 0}

def get_server_ip():
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
    if CUSTOM_DOMAIN:
        protocol = 'https' if USE_HTTPS else 'http'
        return f"{protocol}://{CUSTOM_DOMAIN}"
    
    if request:
        forwarded_proto = request.headers.get('X-Forwarded-Proto')
        forwarded_host = request.headers.get('X-Forwarded-Host')
        
        if forwarded_proto and forwarded_host:
            return f"{forwarded_proto}://{forwarded_host}"
        
        protocol = 'https' if request.is_secure else 'http'
        host = forwarded_host or request.headers.get('Host') or f"{SERVER_IP}:{PORT}"
        return f"{protocol}://{host}"
    
    protocol = 'https' if USE_HTTPS else 'http'
    return f"{protocol}://{SERVER_IP}:{PORT}"

def update_task_progress(task_id, progress, status, message="", result_url="", detailed_info=None):
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
    with task_lock:
        return task_status.get(task_id, {'status': 'not_found'})

def validate_audio(audio_path):
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
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    if len(filename) > 100:
        name, ext = os.path.splitext(filename)
        filename = name[:96] + ext
    return filename

def runffmpeg(arg):
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

def save_audio(file_data, original_filename, task_id):
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
    try:
        update_task_progress(task_id, 60, 'processing', '正在验证音频文件...')
        
        if not validate_audio(audio_path):
            logger.error("输入音频验证失败")
            update_task_progress(task_id, 0, 'failed', '输入音频文件无效')
            return audio_path
        
        update_task_progress(task_id, 70, 'processing', '正在加载降噪模型...')
        ans = gpu_manager.get_model()
        
        update_task_progress(task_id, 80, 'processing', '模型处理中，请稍候...')
        
        progress_capture = RealTimeProgressCapture(task_id, base_progress=80)
        
        class TeeOutput:
            def __init__(self, original, capture):
                self.original = original
                self.capture = capture
            
            def write(self, text):
                result = self.original.write(text)
                self.original.flush()
                self.capture.write(text)
                return result
            
            def flush(self):
                self.original.flush()
                self.capture.flush()
        
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        try:
            sys.stdout = TeeOutput(original_stdout, progress_capture)
            sys.stderr = TeeOutput(original_stderr, progress_capture)
            
            logger.info(f"开始模型处理: {audio_path}")
            result = ans(audio_path, output_path=output_file)
            
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        
        update_task_progress(task_id, 99, 'processing', '正在完成处理...')
        
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            if not base_url:
                base_url = get_base_url()
            
            result_url = f'{base_url}/tmp/{Path(output_file).name}'
            file_size = os.path.getsize(output_file)
            
            detailed_info = {
                'stage': 'completed',
                'file_size': file_size,
                'final_result': True
            }
            
            update_task_progress(
                task_id, 100, 'completed', 
                '降噪处理完成！', result_url, detailed_info
            )
            
            logger.info(f"降噪处理成功: {output_file}, 大小: {file_size} bytes")
            return output_file
        else:
            update_task_progress(task_id, 0, 'failed', '降噪处理失败')
            return audio_path
            
    except Exception as e:
        logger.error(f"降噪处理异常: {e}")
        update_task_progress(task_id, 0, 'failed', f'降噪处理失败: {str(e)}')
        return audio_path

def process_audio_async(converted_path, task_id, base_url=None):
    try:
        noise_removed_path = converted_path.replace('-16kconver.wav', '-remove-noise.wav')
        result_path = remove_noise_with_realtime_progress(converted_path, noise_removed_path, task_id, base_url)
        return result_path
    except Exception as e:
        logger.error(f"异步处理失败: {e}")
        update_task_progress(task_id, 0, 'failed', f'处理失败: {str(e)}')
        return None

def cleanup_old_files():
    try:
        current_time = time.time()
        cleaned_count = 0
        
        for filename in os.listdir(TMPDIR):
            file_path = os.path.join(TMPDIR, filename)
            if os.path.isfile(file_path):
                if current_time - os.path.getctime(file_path) > 3600:
                    os.remove(file_path)
                    cleaned_count += 1
        
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

# ==================== API 路由 ====================

@app.route('/tmp/<path:filename>')
def tmp_files(filename):
    """下载处理后的音频文件"""
    try:
        safe_path = os.path.join(TMPDIR, os.path.basename(filename))
        if os.path.exists(safe_path) and os.path.commonpath([TMPDIR, safe_path]) == TMPDIR:
            return send_file(safe_path, as_attachment=True)
        return jsonify({'code': -1, 'msg': '文件不存在'}), 404
    except Exception as e:
        return jsonify({'code': -1, 'msg': '文件访问失败'}), 500

@app.route('/health')
def health_check():
    """
    健康检查接口
    ---
    tags:
      - 系统管理
    responses:
      200:
        description: 服务健康状态
        schema:
          type: object
          properties:
            status:
              type: string
              example: healthy
            model_loaded:
              type: boolean
            gpu_idle_time:
              type: number
            active_tasks:
              type: integer
    """
    base_url = get_base_url(request)
    return jsonify({
        'status': 'healthy',
        'server_ip': SERVER_IP,
        'port': PORT,
        'custom_domain': CUSTOM_DOMAIN,
        'use_https': USE_HTTPS,
        'base_url': base_url,
        'model_loaded': gpu_manager.is_model_loaded(),
        'gpu_idle_time': int(gpu_manager.get_idle_time()),
        'gpu_idle_timeout': GPU_IDLE_TIMEOUT,
        'active_tasks': len(task_status),
        'timestamp': time.time()
    })

@app.route('/upload_async', methods=['POST'])
def upload_file_async():
    """
    异步上传音频文件进行降噪处理
    ---
    tags:
      - 音频处理
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: 音频文件（支持 MP3、WAV、M4A 等格式）
    responses:
      200:
        description: 上传成功，返回任务ID
        schema:
          type: object
          properties:
            code:
              type: integer
              example: 0
            msg:
              type: string
              example: 文件上传成功，正在处理中
            data:
              type: object
              properties:
                task_id:
                  type: string
                status_url:
                  type: string
                estimated_time:
                  type: string
      400:
        description: 请求错误
    """
    try:
        if 'audio' not in request.files:
            return jsonify({'code': -1, 'msg': '必须选择文件上传'})
        
        file = request.files['audio']
        if file.filename == '':
            return jsonify({'code': -1, 'msg': '未选择上传文件'})
        
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        
        # 记录文件大小用于日志
        logger.info(f"接收到文件: {file.filename}, 大小: {size} bytes ({size/1024/1024:.2f} MB)")
        
        file_data = file.read()
        original_filename = file.filename
        
        base_url = get_base_url(request)
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
    """
    查询任务处理状态
    ---
    tags:
      - 音频处理
    parameters:
      - name: task_id
        in: path
        type: string
        required: true
        description: 任务ID
    responses:
      200:
        description: 任务状态信息
        schema:
          type: object
          properties:
            code:
              type: integer
            data:
              type: object
              properties:
                task_id:
                  type: string
                status:
                  type: string
                  enum: [processing, completed, failed]
                progress:
                  type: integer
                message:
                  type: string
                result_url:
                  type: string
    """
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

@app.route('/api', methods=['GET', 'POST'])
def denoise():
    """
    同步降噪处理接口（向后兼容）
    ---
    tags:
      - 音频处理
    consumes:
      - multipart/form-data
    parameters:
      - name: audio
        in: formData
        type: file
        required: true
        description: 音频文件
      - name: stream
        in: formData
        type: integer
        default: 0
        description: 是否直接返回音频流（0=返回URL，1=返回文件）
    responses:
      200:
        description: 处理成功
    """
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

@app.route('/gpu/status')
def gpu_status():
    """
    GPU 资源状态
    ---
    tags:
      - 系统管理
    responses:
      200:
        description: GPU 状态信息
    """
    return jsonify({
        'model_loaded': gpu_manager.is_model_loaded(),
        'idle_time': int(gpu_manager.get_idle_time()),
        'idle_timeout': GPU_IDLE_TIMEOUT,
        'will_release_in': max(0, GPU_IDLE_TIMEOUT - int(gpu_manager.get_idle_time()))
    })

@app.route('/')
def index():
    """主页 - 音频降噪 Web UI"""
    base_url = get_base_url(request)
    gpu_timeout_minutes = GPU_IDLE_TIMEOUT // 60
    
    with open('ui_template.html', 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    html_content = html_content.replace('__BASE_URL__', base_url)
    html_content = html_content.replace('__GPU_TIMEOUT__', str(gpu_timeout_minutes))
    
    return Response(html_content, mimetype='text/html')

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
        cleanup_old_files()
        
        def periodic_cleanup():
            while True:
                time.sleep(3600)
                cleanup_old_files()
        
        threading.Thread(target=periodic_cleanup, daemon=True).start()
        
        print(f"🚀 音频降噪服务启动成功！")
        print(f"📱 本地访问: http://127.0.0.1:{PORT}")
        print(f"📚 API 文档: http://127.0.0.1:{PORT}/docs")
        print(f"💊 健康检查: http://127.0.0.1:{PORT}/health")
        print(f"🎮 GPU 状态: http://127.0.0.1:{PORT}/gpu/status")
        
        if CUSTOM_DOMAIN:
            protocol = 'https' if USE_HTTPS else 'http'
            print(f"🌐 公网访问: {protocol}://{CUSTOM_DOMAIN}")
            print(f"📚 公网文档: {protocol}://{CUSTOM_DOMAIN}/docs")
        else:
            print(f"🌐 公网访问: http://{SERVER_IP}:{PORT}")
        
        print(f"📊 服务器IP: {SERVER_IP}")
        print(f"🔧 自定义域名: {CUSTOM_DOMAIN or '未配置'}")
        print(f"🔒 HTTPS模式: {'启用' if USE_HTTPS else '禁用'}")
        print(f"🎮 GPU 空闲超时: {GPU_IDLE_TIMEOUT // 60} 分钟")
        
        threading.Thread(target=openweb, args=(f'http://127.0.0.1:{PORT}',)).start()
        
        serve(app, host=HOST, port=PORT, threads=8)
        
    except Exception as e:
        logger.error(f"服务启动失败: {str(e)}")
        logger.error(traceback.format_exc())
    finally:
        gpu_manager.stop()
