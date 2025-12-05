# 🎵 オーディオノイズ除去サービス

[English](README.md) | [简体中文](README_CN.md) | [繁體中文](README_TW.md) | [日本語](README_JP.md)

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)
[![CUDA](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

> AI駆動のオーディオノイズ除去サービス。自動GPU管理、リアルタイム進捗追跡、包括的なAPIドキュメントを提供。

---

## ✨ 機能

- 🎯 **AI駆動ノイズ除去**：ModelScope ZipEnhancerモデルベース
- 🎮 **スマートGPU管理**：最も空いているGPUを自動選択、アイドル時に自動解放
- 🐳 **Docker対応**：ワンコマンドデプロイ、完全なGPUサポート
- 📚 **Swagger APIドキュメント**：`/docs`でインタラクティブなAPIドキュメント
- 🌐 **デュアルモード**：モダンなWeb UI + RESTful API
- ⚡ **リアルタイム進捗**：ETAと処理速度を表示するライブ進捗バー
- 🔄 **自動クリーンアップ**：1時間後に一時ファイルを自動削除
- 🌍 **多言語対応**：英語、中国語（簡体字/繁体字）、日本語

---

## 🚀 クイックスタート

### オプション1：Docker（推奨）

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# サービスを起動（最適なGPUを自動選択）
./start.sh

# サービスにアクセス
# Web UI: http://0.0.0.0:5080
# APIドキュメント: http://0.0.0.0:5080/docs
```

### オプション2：直接実行

```bash
# 依存関係をインストール
pip install -r requirements.txt --no-deps
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# システム依存関係をインストール（Ubuntu/Debian）
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1

# サービスを起動
python api_enhanced.py

# アクセス: http://127.0.0.1:5080
```

---

## 📦 インストール

### 前提条件

- **Docker**: 20.10+（Dockerデプロイ用）
- **Docker Compose**: 1.29+
- **NVIDIA Docker**: nvidia-docker2
- **GPU**: NVIDIA GPU、VRAM 4GB以上
- **Python**: 3.10+（直接実行用）
- **CUDA**: 12.1+（GPUアクセラレーション用）

### Dockerインストール

```bash
# 1. リポジトリをクローン
git clone https://github.com/yourusername/remove-noise-service.git
cd remove-noise-service

# 2. 設定（オプション）
cp .env.example .env
nano .env

# 3. サービスを起動
./start.sh
```

---

## ⚙️ 設定

### 環境変数

| 変数 | 説明 | デフォルト | 例 |
|------|------|-----------|-----|
| `PORT` | サービスポート | 5080 | 5080 |
| `CUSTOM_DOMAIN` | カスタムドメイン | - | noise.example.com |
| `USE_HTTPS` | HTTPSを使用 | true | true/false |
| `GPU_IDLE_TIMEOUT` | GPUアイドルタイムアウト（分） | 10 | 10 |
| `GPU_ID` | GPU ID（自動選択） | 0 | 0, 1, 2... |

---

## 💻 使用方法

### Web UI

1. ブラウザを開く：http://0.0.0.0:5080
2. オーディオファイルをドラッグ＆ドロップまたはクリックして選択
3. 処理を待つ（リアルタイム進捗表示）
4. 結果をダウンロード

### API使用

#### 非同期処理（推奨）

```bash
# 1. ファイルをアップロード
curl -X POST http://localhost:5080/upload_async \
  -F "audio=@your_audio.mp3"

# 2. ステータスを確認
curl http://localhost:5080/status/<task_id>
```

---

## 📚 APIドキュメント

完全なSwaggerドキュメントは http://0.0.0.0:5080/docs をご覧ください。

---

## 🔧 よく使うコマンド

```bash
make help      # すべてのコマンドを表示
make start     # サービスを起動
make stop      # サービスを停止
make logs      # ログを表示
make test      # テストを実行
```

---

## 📝 変更履歴

### v2.0.0 (2025-12-05)
- ✨ 完全なDockerデプロイ
- ✨ 自動GPU選択と管理
- ✨ Swagger APIドキュメント
- ✨ 強化されたWeb UI
- ✨ リアルタイム進捗追跡

---

## 📄 ライセンス

このプロジェクトはMITライセンスの下でライセンスされています - 詳細は[LICENSE](LICENSE)ファイルをご覧ください。

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/remove-noise-service&type=Date)](https://star-history.com/#yourusername/remove-noise-service)

---

## 📱 フォローする

![WeChat公式アカウント](https://img.aws.xin/uPic/扫码_搜索联合传播样式-标准色版.png)

**「AI健自習室」をフォローして、より多くのAIツールとチュートリアルを入手**
