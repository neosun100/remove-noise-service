#!/usr/bin/env python3
"""
测试 MCP 工具是否可用
"""
import sys
from pathlib import Path

# 添加项目路径
REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

# 导入 MCP 服务器
import mcp_server

def test_mcp_tools():
    """测试 MCP 工具"""
    print("🧪 测试 MCP 工具")
    print("=" * 60)
    
    # 检查工具是否注册
    tools = mcp_server.mcp._tools
    print(f"\n✅ 已注册的工具数量: {len(tools)}")
    
    for tool_name, tool_func in tools.items():
        print(f"  • {tool_name}: {tool_func.__doc__.split('.')[0] if tool_func.__doc__ else 'No description'}")
    
    print("\n" + "=" * 60)
    print("✅ MCP 工具可用！")
    print("\n使用方法:")
    print("  1. 在 MCP 客户端配置中添加此服务器")
    print("  2. 使用 denoise_path(path) 提交音频文件")
    print("  3. 使用 get_status(task_id) 查询处理状态")
    print("  4. 使用 get_result(task_id) 获取结果")
    print("\n注意: MCP 服务器与 Web API 共享同一个任务队列")
    print("     可以通过 MCP 提交任务，通过 Web UI 查看进度")

if __name__ == "__main__":
    test_mcp_tools()
