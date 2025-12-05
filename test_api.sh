#!/bin/bash

set -e

echo "🧪 音频降噪服务 API 测试"
echo "=========================="

BASE_URL="${1:-http://localhost:5080}"

echo ""
echo "1️⃣ 测试健康检查..."
curl -s "$BASE_URL/health" | python3 -m json.tool

echo ""
echo ""
echo "2️⃣ 测试 GPU 状态..."
curl -s "$BASE_URL/gpu/status" | python3 -m json.tool

echo ""
echo ""
echo "3️⃣ 测试 Swagger 文档..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/docs")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ Swagger 文档可访问: $BASE_URL/docs"
else
    echo "❌ Swagger 文档不可访问 (HTTP $HTTP_CODE)"
fi

echo ""
echo ""
echo "4️⃣ 测试异步上传 API..."
if [ -f "test_audio.wav" ]; then
    echo "使用测试文件: test_audio.wav"
    RESPONSE=$(curl -s -F "audio=@test_audio.wav" "$BASE_URL/upload_async")
    echo "$RESPONSE" | python3 -m json.tool
    
    TASK_ID=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['task_id'])" 2>/dev/null || echo "")
    
    if [ -n "$TASK_ID" ]; then
        echo ""
        echo "任务ID: $TASK_ID"
        echo "正在查询状态..."
        
        for i in {1..10}; do
            sleep 2
            STATUS=$(curl -s "$BASE_URL/status/$TASK_ID")
            echo "$STATUS" | python3 -m json.tool
            
            TASK_STATUS=$(echo "$STATUS" | python3 -c "import sys, json; print(json.load(sys.stdin)['data']['status'])" 2>/dev/null || echo "")
            
            if [ "$TASK_STATUS" = "completed" ]; then
                echo ""
                echo "✅ 任务完成！"
                break
            elif [ "$TASK_STATUS" = "failed" ]; then
                echo ""
                echo "❌ 任务失败"
                break
            fi
        done
    fi
else
    echo "⚠️  未找到测试文件 test_audio.wav，跳过上传测试"
    echo "   提示: 将音频文件命名为 test_audio.wav 放在当前目录即可测试"
fi

echo ""
echo ""
echo "=========================="
echo "✅ 测试完成！"
echo ""
echo "📚 访问 Swagger 文档: $BASE_URL/docs"
echo "🌐 访问 Web UI: $BASE_URL/"
