"""
10_mcp_weather_server.py — MCP 服务端（MCP SDK 1.x）
提供天气查询 + 计算器 + 时间 三个工具
"""

import asyncio
import math
from datetime import datetime
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("weather-calc-service")

WEATHER_DB = {
    "贵阳": "凉爽，22°C，多云转晴",
    "北京": "炎热，35°C，晴天",
    "上海": "闷热，31°C，阵雨",
    "广州": "热，33°C，雷阵雨",
    "深圳": "热，32°C，多云",
    "成都": "舒适，26°C，阴天",
}


# ============ MCP 1.x 的工具注册方式 ============
# 方式：用 @server.list_tools() 声明有哪些工具
#      用 @server.call_tool() 处理工具调用

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    """告诉客户端：我有哪些工具、各自需要什么参数"""
    return [
        Tool(
            name="get_weather",
            description="查询指定城市的天气",
            inputSchema={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如'贵阳'、'北京'"}
                },
                "required": ["city"],
            },
        ),
        Tool(
            name="calculate",
            description="计算数学表达式",
            inputSchema={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如'3*5+2'"}
                },
                "required": ["expression"],
            },
        ),
        Tool(
            name="get_current_time",
            description="获取当前日期和时间",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    """执行具体的工具调用，返回结果"""
    if name == "get_weather":
        city = arguments["city"]
        result = WEATHER_DB.get(city, f"抱歉，没有{city}的天气数据")

    elif name == "calculate":
        expr = arguments["expression"]
        try:
            result = f"{expr} = {eval(expr, {'__builtins__': {}}, {'sqrt': math.sqrt})}"
        except Exception as e:
            result = f"计算出错：{e}"

    elif name == "get_current_time":
        result = datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")

    else:
        result = f"未知工具：{name}"

    return [TextContent(type="text", text=result)]


async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
