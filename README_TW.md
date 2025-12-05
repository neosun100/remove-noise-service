# 🎵 音訊降噪服務

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> 基於 AI 的音訊降噪服務，支援自動 GPU 管理、即時進度追蹤和完整的 API 文件。

---

## ✨ 功能特性

- 🎯 **AI 驅動降噪**：基於 ModelScope ZipEnhancer 模型
- 🎮 **智慧 GPU 管理**：自動選擇最空閒 GPU，空閒時自動釋放
- 🐳 **Docker 就緒**：一鍵部署，完整 GPU 支援
- 📚 **Swagger API 文件**：互動式 API 文件，訪問 `/docs`
- 🌐 **雙模式**：現代化 Web UI + RESTful API
- ⚡ **即時進度**：即時進度條，顯示 ETA 和處理速度
- 🔄 **自動清理**：臨時檔案 1 小時後自動清理
- 🌍 **多語言**：英文、簡體中文、繁體中文、日文

### 📸 Web UI 預覽

![Web UI 截圖](https://img.aws.xin/uPic/RJZXJa.png)

*現代化 Web 介面，支援拖曳上傳、即時進度追蹤和即時下載*

---

## 🚀 快速開始

### 方式一：Docker（推薦）

```bash
# 克隆倉庫
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 啟動服務（自動選擇最佳 GPU）
./start.sh

# 訪問服務
# Web UI: http://0.0.0.0:5080
# API 文件: http://0.0.0.0:5080/docs
```

### 方式二：直接執行

```bash
# 安裝依賴
pip install -r requirements.txt --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安裝系統依賴（Ubuntu/Debian）
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# 啟動服務
python api_enhanced.py

# 訪問: http://127.0.0.1:5080
```

---

## 📦 安裝部署

### 前置要求

- **Docker**: 20.10+（Docker 部署）
- **Docker Compose**: 1.29+
- **NVIDIA Docker**: nvidia-docker2
- **GPU**: NVIDIA GPU，顯存 4GB+
- **Python**: 3.10+（直接執行）
- **CUDA**: 12.1+（GPU 加速）

### Docker 安裝

```bash
# 1. 克隆倉庫
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 2. 配置（可選）
cp .env.example .env
nano .env

# 3. 啟動服務
./start.sh
```

---

## ⚙️ 配置說明

### 環境變數

| 變數 | 說明 | 預設值 | 範例 |
|------|------|--------|------|
| `PORT` | 服務埠 | 5080 | 5080 |
| `CUSTOM_DOMAIN` | 自訂網域 | - | noise.example.com |
| `USE_HTTPS` | 使用 HTTPS | true | true/false |
| `GPU_IDLE_TIMEOUT` | GPU 空閒逾時（分鐘） | 10 | 10 |
| `GPU_ID` | GPU ID（自動選擇） | 0 | 0, 1, 2... |

---

## 💻 使用方法

### Web UI

1. 開啟瀏覽器：http://0.0.0.0:5080
2. 拖曳音訊檔案或點擊選擇
3. 等待處理（顯示即時進度）
4. 下載結果

### API 使用

#### 非同步處理（推薦）

```bash
# 1. 上傳檔案
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# 2. 查詢狀態
curl http://localhost:5080/status/<task_id>
```

---

## 📚 API 文件

訪問 http://0.0.0.0:5080/docs 查看完整的 Swagger 文件。

---

## 🔧 常用命令

```bash
make help      # 顯示所有命令
make start     # 啟動服務
make stop      # 停止服務
make logs      # 查看日誌
make test      # 執行測試
```

---

## 📝 更新日誌

### v2.0.0 (2025-12-05)
- ✨ 完整 Docker 部署
- ✨ 自動 GPU 選擇和管理
- ✨ Swagger API 文件
- ✨ 增強的 Web UI
- ✨ 即時進度追蹤

---

## 📄 授權

本專案採用 MIT 授權 - 詳見 [LICENSE](LICENSE) 檔案。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/remove-noise-service&type=Date)](https://star-history.com/#yourusername/remove-noise-service)

---

## 📱 關注我們

![微信公眾號](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

**掃碼關注「AI健自習室」獲取更多 AI 工具和教程**
