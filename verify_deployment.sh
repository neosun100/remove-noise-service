#!/bin/bash

echo "🔍 验证部署文件完整性"
echo "======================="
echo ""

REQUIRED_FILES=(
    "Dockerfile"
    "docker-compose.yml"
    ".env.example"
    ".dockerignore"
    "start.sh"
    "test_api.sh"
    "api_enhanced.py"
    "gpu_manager.py"
    "ui_template.html"
    "requirements.txt"
    "README_DOCKER.md"
    "QUICKSTART.md"
    "DEPLOYMENT_CHECKLIST.md"
    "PROJECT_SUMMARY.md"
    "Makefile"
)

MISSING=0
PRESENT=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file"
        PRESENT=$((PRESENT + 1))
    else
        echo "❌ $file (缺失)"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
echo "======================="
echo "总计: $((PRESENT + MISSING)) 个文件"
echo "存在: $PRESENT 个"
echo "缺失: $MISSING 个"
echo ""

if [ $MISSING -eq 0 ]; then
    echo "✅ 所有文件完整！"
    echo ""
    echo "下一步:"
    echo "  1. 运行: ./start.sh"
    echo "  2. 访问: http://0.0.0.0:5080"
    echo "  3. 测试: ./test_api.sh"
    exit 0
else
    echo "❌ 有文件缺失，请检查！"
    exit 1
fi
