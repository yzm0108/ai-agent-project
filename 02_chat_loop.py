"""
02_chat_loop.py — 多轮对话
功能：在终端里跟 AI 一直聊，直到你输入 quit 退出
这是 Agent 的基础——Agent 需要跟 LLM 来回对话才能完成任务
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# messages 是一个列表，用来存储整个对话历史
# AI 本身记不住上一句话，全靠你把历史消息一次性发过去
# 所以你每次都要把之前的所有对话重新发一遍
messages = [
    {"role": "system", "content": "你是一个友好的助手，用中文回答。"},
]

print("=" * 50)
print("AI 对话模式已启动！输入 quit 退出")
print("=" * 50)

while True:
    # 第 1 步：读取你输入的话
    user_input = input("\n你: ")

    # 第 2 步：如果你输入 quit，退出循环
    if user_input.strip().lower() == "quit":
        print("再见！")
        break

    # 第 3 步：把你输入的话追加到消息列表
    messages.append({"role": "user", "content": user_input})

    # 第 4 步：把整个消息列表发给 DeepSeek
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )

    # 第 5 步：取出 AI 回复
    ai_reply = response.choices[0].message.content

    # 第 6 步：打印 AI 回复
    print(f"AI: {ai_reply}")

    # 第 7 步：把 AI 回复也追加到消息列表（否则下次 AI 就不记得自己说过什么了）
    messages.append({"role": "assistant", "content": ai_reply})
