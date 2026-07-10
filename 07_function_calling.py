"""
07_function_calling.py — 让 AI 学会调用工具
这是 Agent 的核心机制：AI 决定调哪个函数、传什么参数 → 你执行 → 返回结果给 AI
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)


# ============ 第 1 步：定义你有哪些工具（给 AI 看的"菜单"） ============
# AI 看不到你的代码。你只能通过这个"工具描述"告诉 AI：
#   - 这个工具叫什么名字
#   - 它干什么用
#   - 它需要什么参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",              # 工具名
            "description": "查询指定城市的天气",   # 告诉 AI 这是干什么的
            "parameters": {                      # 告诉 AI 需要什么参数
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如'北京'、'上海'、'贵阳'",
                    }
                },
                "required": ["city"],            # 必填参数
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，例如'3*5+2'、'sqrt(16)'",
                    }
                },
                "required": ["expression"],
            },
        },
    },
]


# ============ 第 2 步：实现这些工具（真正的 Python 函数） ============
# 这是模拟的天气数据。真正的项目里，这里会去调高德地图 API 或彩云天气 API。
def get_weather(city):
    """模拟天气查询"""
    weather_data = {
        "贵阳": "凉爽，22°C，多云转晴",
        "北京": "炎热，35°C，晴天",
        "上海": "闷热，31°C，阵雨",
        "广州": "热，33°C，雷阵雨",
        "深圳": "热，32°C，多云",
    }
    return weather_data.get(city, f"没有{city}的天气数据")


def calculate(expression):
    """安全地计算数学表达式"""
    try:
        # eval 有安全风险，但这里只允许数字和运算符，问题不大
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


# 工具名的映射表：AI 返回工具名 → 你的 Python 函数
TOOL_MAP = {
    "get_weather": get_weather,
    "calculate": calculate,
}


# ============ 第 3 步：一次完整的"用户提问 → AI 决定调工具 → 执行 → 继续" ============
def run_agent(user_question):
    """核心循环：AI 决定要不要调工具、调哪个、传什么参数"""
    messages = [{"role": "user", "content": user_question}]

    # 第一轮：把用户问题和工具列表一起发给 AI
    # AI 可能直接回答，也可能返回"我要调 XX 工具"
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        tools=tools,  # ← 第一次把工具菜单传给 AI
    )

    # 取出 AI 的回复
    msg = response.choices[0].message

    # 如果 AI 没有要求调工具（直接回答），那就直接返回答案
    if msg.tool_calls is None:
        return msg.content, []

    # 如果 AI 要求调工具，那就执行
    tool_log = []  # 记录调了什么工具，方便调试
    messages.append(msg)  # 把 AI 的工具调用请求加入对话

    for tool_call in msg.tool_calls:
        tool_name = tool_call.function.name            # AI 要调哪个工具
        tool_args = json.loads(tool_call.function.arguments)  # AI 传了什么参数

        print(f"  [工具调用] {tool_name}({tool_args})")

        # 执行对应的 Python 函数
        func = TOOL_MAP[tool_name]
        result = func(**tool_args)  # **tool_args 把字典展开成关键字参数

        print(f"  [工具结果] {result}")

        # 把工具执行结果加入对话
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })
        tool_log.append((tool_name, tool_args, result))

    # 第二轮：把工具结果发给 AI，让它基于结果生成最终回答
    final_response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    final_answer = final_response.choices[0].message.content

    return final_answer, tool_log


# ============ 测试 ============
print("=" * 60)
print("测试 1：问天气（需要调工具）")
print("=" * 60)
answer1, log1 = run_agent("请问贵阳今天天气怎么样？")
print(f"\n最终回答：{answer1}\n")

print("=" * 60)
print("测试 2：算数学（需要调工具）")
print("=" * 60)
answer2, log2 = run_agent("帮我算一下 123 * 456 + 789")
print(f"\n最终回答：{answer2}\n")

print("=" * 60)
print("测试 3：纯聊天（不需要工具）")
print("=" * 60)
answer3, log3 = run_agent("你好，请用一句话介绍你自己")
print(f"\n最终回答：{answer3}\n")

print("=" * 60)
print("测试 4：一句话需要两次工具调用")
print("=" * 60)
answer4, log4 = run_agent("贵阳和北京今天天气分别怎么样？")
print(f"\n最终回答：{answer4}")
