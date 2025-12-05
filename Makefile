.PHONY: help start stop restart logs build test clean status health gpu

help:
	@echo "🎵 音频降噪服务 - 快捷命令"
	@echo ""
	@echo "使用方法: make [命令]"
	@echo ""
	@echo "常用命令:"
	@echo "  start      - 启动服务（自动选择 GPU）"
	@echo "  stop       - 停止服务"
	@echo "  restart    - 重启服务"
	@echo "  logs       - 查看日志"
	@echo "  build      - 重新构建镜像"
	@echo "  test       - 运行测试"
	@echo "  clean      - 清理容器和镜像"
	@echo "  status     - 查看服务状态"
	@echo "  health     - 健康检查"
	@echo "  gpu        - 查看 GPU 状态"
	@echo ""

start:
	@echo "🚀 启动服务..."
	@./start.sh

stop:
	@echo "🛑 停止服务..."
	@docker-compose down

restart:
	@echo "🔄 重启服务..."
	@docker-compose restart

logs:
	@echo "📋 查看日志..."
	@docker-compose logs -f

build:
	@echo "🔨 重新构建镜像..."
	@docker-compose build --no-cache
	@docker-compose up -d

test:
	@echo "🧪 运行测试..."
	@./test_api.sh

clean:
	@echo "🧹 清理容器和镜像..."
	@docker-compose down -v
	@docker image prune -f

status:
	@echo "📊 服务状态:"
	@docker ps | grep remove-noise-service || echo "服务未运行"
	@echo ""
	@echo "📈 资源使用:"
	@docker stats remove-noise-service --no-stream 2>/dev/null || echo "无法获取资源信息"

health:
	@echo "💊 健康检查:"
	@curl -s http://localhost:5080/health | python3 -m json.tool || echo "服务不可访问"

gpu:
	@echo "🎮 GPU 状态:"
	@curl -s http://localhost:5080/gpu/status | python3 -m json.tool || echo "无法获取 GPU 状态"
	@echo ""
	@echo "🖥️  系统 GPU 信息:"
	@nvidia-smi || echo "无法获取 GPU 信息"
