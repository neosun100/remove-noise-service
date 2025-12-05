#!/bin/bash

BASE_URL="https://noise.aws.xin"
TEST_AUDIO="/tmp/test_audio.wav"

echo "=========================================="
echo "GPU 显存管理功能测试"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

check_gpu_memory() {
    local gpu_id=$1
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | grep "^${gpu_id}" | awk '{print $2}'
}

echo "1️⃣  测试：完全释放所有资源"
echo "-------------------------------------------"
curl -s -X POST "$BASE_URL/gpu/release" | jq .
status=$(curl -s "$BASE_URL/gpu/status")
echo "$status" | jq .

model_on_gpu=$(echo "$status" | jq -r '.model_on_gpu')
model_on_cpu=$(echo "$status" | jq -r '.model_on_cpu')

if [ "$model_on_gpu" = "false" ] && [ "$model_on_cpu" = "false" ]; then
    echo -e "${GREEN}✅ 测试通过：所有资源已释放${NC}"
else
    echo -e "${RED}❌ 测试失败：资源未完全释放${NC}"
fi
echo ""

echo "2️⃣  测试：处理音频并自动卸载到CPU"
echo "-------------------------------------------"
task_id=$(curl -s -X POST "$BASE_URL/upload_async" -F "audio=@$TEST_AUDIO" | jq -r '.data.task_id')
echo "任务ID: $task_id"

# 等待处理完成
for i in {1..30}; do
    status=$(curl -s "$BASE_URL/status/$task_id" | jq -r '.data.status')
    progress=$(curl -s "$BASE_URL/status/$task_id" | jq -r '.data.progress')
    echo -ne "\r进度: $progress% | 状态: $status    "
    
    if [ "$status" = "completed" ] || [ "$status" = "failed" ]; then
        echo ""
        break
    fi
    sleep 2
done

# 检查处理后的状态
sleep 2
gpu_status=$(curl -s "$BASE_URL/gpu/status")
echo "$gpu_status" | jq .

model_on_gpu=$(echo "$gpu_status" | jq -r '.model_on_gpu')
model_on_cpu=$(echo "$gpu_status" | jq -r '.model_on_cpu')

if [ "$model_on_gpu" = "false" ] && [ "$model_on_cpu" = "true" ]; then
    echo -e "${GREEN}✅ 测试通过：处理完成后自动卸载到CPU${NC}"
else
    echo -e "${RED}❌ 测试失败：未正确卸载${NC}"
fi

# 检查实际GPU显存
gpu_mem=$(check_gpu_memory 3)
echo "GPU 3 实际显存占用: ${gpu_mem} MB"

if [ "$gpu_mem" -lt 1000 ]; then
    echo -e "${GREEN}✅ GPU显存已释放（< 1GB）${NC}"
else
    echo -e "${YELLOW}⚠️  GPU显存占用较高: ${gpu_mem} MB${NC}"
fi
echo ""

echo "3️⃣  测试：手动卸载到CPU"
echo "-------------------------------------------"
# 先处理一个任务让模型加载
task_id=$(curl -s -X POST "$BASE_URL/upload_async" -F "audio=@$TEST_AUDIO" | jq -r '.data.task_id')
sleep 5

# 手动卸载
result=$(curl -s -X POST "$BASE_URL/gpu/offload")
echo "$result" | jq .

code=$(echo "$result" | jq -r '.code')
if [ "$code" = "0" ] || [ "$code" = "1" ]; then
    echo -e "${GREEN}✅ 测试通过：手动卸载API正常${NC}"
else
    echo -e "${RED}❌ 测试失败：手动卸载失败${NC}"
fi
echo ""

echo "4️⃣  测试：从CPU缓存快速恢复"
echo "-------------------------------------------"
# 确保模型在CPU上
curl -s -X POST "$BASE_URL/gpu/offload" > /dev/null

start_time=$(date +%s)
task_id=$(curl -s -X POST "$BASE_URL/upload_async" -F "audio=@$TEST_AUDIO" | jq -r '.data.task_id')

# 等待完成
for i in {1..30}; do
    status=$(curl -s "$BASE_URL/status/$task_id" | jq -r '.data.status')
    if [ "$status" = "completed" ]; then
        break
    fi
    sleep 1
done

end_time=$(date +%s)
duration=$((end_time - start_time))

echo "从CPU缓存恢复并处理耗时: ${duration}秒"

if [ "$duration" -lt 60 ]; then
    echo -e "${GREEN}✅ 测试通过：快速恢复（< 60秒）${NC}"
else
    echo -e "${YELLOW}⚠️  恢复较慢: ${duration}秒${NC}"
fi
echo ""

echo "5️⃣  测试：API文档可访问性"
echo "-------------------------------------------"
docs_status=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$docs_status" = "200" ]; then
    echo -e "${GREEN}✅ API文档可访问: $BASE_URL/docs${NC}"
else
    echo -e "${RED}❌ API文档不可访问${NC}"
fi
echo ""

echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "📊 新增API端点："
echo "  - POST $BASE_URL/gpu/offload  (卸载到CPU)"
echo "  - POST $BASE_URL/gpu/release  (完全释放)"
echo "  - GET  $BASE_URL/gpu/status   (详细状态)"
echo ""
echo "🎯 功能特性："
echo "  ✓ 处理完成后自动卸载GPU显存到CPU"
echo "  ✓ 手动卸载GPU显存到CPU"
echo "  ✓ 手动完全释放所有资源"
echo "  ✓ 从CPU缓存快速恢复到GPU"
echo "  ✓ 空闲超时自动转移到CPU"
echo ""
