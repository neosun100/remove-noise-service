#!/bin/bash

set -e

echo "🚀 音频降噪服务 Docker 启动脚本"
echo "================================"

# 检查 nvidia-docker
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 错误: 未检测到 nvidia-smi，请确保已安装 NVIDIA 驱动"
    exit 1
fi

# 检查 docker
if ! command -v docker &> /dev/null; then
    echo "❌ 错误: 未检测到 docker"
    exit 1
fi

# 检查 docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ 错误: 未检测到 docker-compose"
    exit 1
fi

# 创建 .env 文件（如果不存在）
if [ ! -f .env ]; then
    echo "📝 创建 .env 文件..."
    cp .env.example .env
fi

# 选择显存占用最少的 GPU
echo "🔍 正在检测可用 GPU..."
GPU_INFO=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)

if [ -z "$GPU_INFO" ]; then
    echo "❌ 错误: 未检测到可用 GPU"
    exit 1
fi

echo "GPU 信息:"
echo "$GPU_INFO" | awk '{printf "  GPU %s: 已用 %s MB / 总共 %s MB (%.1f%%)\n", $1, $2, $3, ($2/$3)*100}'

# 计算每个 GPU 的使用率并选择最空闲的
BEST_GPU=$(echo "$GPU_INFO" | awk -F',' '
BEGIN { min_usage = 100; best_gpu = 0 }
{
    usage = ($2 / $3) * 100
    if (usage < min_usage) {
        min_usage = usage
        best_gpu = $1
    }
}
END { print best_gpu }
')

echo ""
echo "✅ 选择 GPU $BEST_GPU (显存占用最少)"

# 更新 .env 文件中的 GPU_ID
if grep -q "^GPU_ID=" .env; then
    sed -i "s/^GPU_ID=.*/GPU_ID=$BEST_GPU/" .env
else
    echo "GPU_ID=$BEST_GPU" >> .env
fi

# 停止旧容器（如果存在）
if [ "$(docker ps -aq -f name=remove-noise-service)" ]; then
    echo "🛑 停止旧容器..."
    docker-compose down
fi

# 构建并启动
echo "🔨 构建 Docker 镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 5

# 检查服务状态
if docker ps | grep -q remove-noise-service; then
    echo ""
    echo "✅ 服务启动成功！"
    echo ""
    echo "📊 服务信息:"
    echo "  - 容器名称: remove-noise-service"
    echo "  - 使用 GPU: $BEST_GPU"
    
    # 读取端口
    PORT=$(grep "^PORT=" .env | cut -d'=' -f2)
    PORT=${PORT:-5080}
    
    echo "  - 访问地址: http://0.0.0.0:$PORT"
    echo "  - API 文档: http://0.0.0.0:$PORT/docs"
    echo "  - 健康检查: http://0.0.0.0:$PORT/health"
    echo ""
    echo "📝 查看日志: docker-compose logs -f"
    echo "🛑 停止服务: docker-compose down"
else
    echo "❌ 服务启动失败，请查看日志: docker-compose logs"
    exit 1
fi
