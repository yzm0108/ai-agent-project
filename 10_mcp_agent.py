"""
10_mcp_agent.py — 通过 MCP 连接工具的 Agent
跟 07-09 的区别：工具不再硬编码在 Agent 里，而是通过 MCP 协议从外部服务获取
架构：Agent ←→ MCP Client ←→ MCP Server（独立进程）
"""

import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

load_dotenv()

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


async def run_mcp_agent(user_question, verbose=True):
    """
    核心流程：
    1. 启动 MCP 服务器子进程
    2. 连接服务器，获取工具列表
    3. 把 MCP 工具 → OpenAI 格式
    4. ReAct 循环：AI 决定调工具 → 通过 MCP 执行 → 观察结果 → 直到回答
    """

    # ===== 第 1 步：启动 MCP 服务器并连接 =====
    # StdioServerParameters 告诉客户端：通过 python 命令启动这个脚本
    server_params = StdioServerParameters(
        command="python",
        args=["10_mcp_weather_server.py"],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 初始化 MCP 会话
            await session.initialize()

            # ===== 第 2 步：从 MCP 服务器获取工具列表 =====
            mcp_tools = await session.list_tools()

            if verbose:
                print(f"已连接 MCP 服务器，发现 {len(mcp_tools.tools)} 个工具：")
                for t in mcp_tools.tools:
                    print(f"  - {t.name}: {t.description}")

            # 把 MCP 工具格式 → OpenAI Function Calling 格式
            openai_tools = []
            for t in mcp_tools.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description or "",
                        "parameters": t.inputSchema if t.inputSchema else {"type": "object", "properties": {}},
                    },
                })

            # ===== 第 3 步：ReAct 循环 =====
            messages = [
                {"role": "system", "content": "你是一个智能助手，可以调用工具获取信息。用中文回答。"},
                {"role": "user", "content": user_question},
            ]

            max_steps = 5
            for step in range(1, max_steps + 1):
                response = deepseek.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    tools=openai_tools,
                )
                msg = response.choices[0].message

                # 不调工具 → 回答完毕
                if msg.tool_calls is None:
                    if verbose:
                        print(f"  [步骤{step}] AI 直接回答")
                    return msg.content

                # AI 要调工具 → 通过 MCP 执行
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    if verbose:
                        print(f"  [步骤{step}] MCP调用：{name}({args})")

                    # ★ 关键行：通过 MCP session 调用服务器上的工具
                    result = await session.call_tool(name, args)

                    if verbose:
                        # result.content 是工具返回的内容列表
                        preview = str(result.content)[:80]
                        print(f"  [步骤{step}] MCP结果：{preview}...")

                    # 把工具调用和结果加入对话
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
                        }],
                    })
                    # 提取文本内容
                    tool_result_text = ""
                    for c in result.content:
                        if hasattr(c, "text"):
                            tool_result_text += c.text
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_result_text,
                    })

            # 超出步数
            messages.append({"role": "user", "content": "请基于已有信息回答。"})
            final = deepseek.chat.completions.create(model="deepseek-chat", messages=messages)
            return final.choices[0].message.content


# ============ 运行 ============
async def main():
    questions = [
        "贵阳今天天气怎么样？",
        "帮我算一下 123 * 456 + 789",
        "现在几点了？顺便告诉我北京天气",
    ]
    for q in questions:
        print("=" * 60)
        print(f"用户：{q}")
        print("-" * 60)
        try:
            answer = await run_mcp_agent(q)
            print(f"\nAI：{answer}\n")
        except Exception as e:
            print(f"出错：{e}\n")


if __name__ == "__main__":
    asyncio.run(main())
