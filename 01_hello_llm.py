"""
01_hello_llm.py — 你的第一个 AI 程序
功能：向 DeepSeek 发送一句话，让它回复你
这是 AI Agent 开发的最底层——调用大模型 API
"""

# 第 1 步：导入需要的库
import os                          # 用来读取系统环境变量
from dotenv import load_dotenv     # 用来读取 .env 文件里的密码
from openai import OpenAI          # OpenAI 的库，DeepSeek 兼容它

# 第 2 步：加载 .env 文件里的 API Key（防止密码泄露到网上）
load_dotenv()

# 第 3 步：创建客户端，连接 DeepSeek 服务器
# 通俗理解：你拿着 API Key 去敲 DeepSeek 的门，说"我是付费用户，让我用你的模型"
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),       # 从 .env 里读取你的 key
    base_url=os.getenv("DEEPSEEK_BASE_URL"),     # DeepSeek 的服务器地址
)

# 第 4 步：发送消息，等它回复
# model 参数：指定用哪个模型，deepseek-chat 是 DeepSeek 的对话模型
# messages：对话内容，system 是给 AI 设定角色，user 是你的问题
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个友好的助手，用中文回答。"},
        {"role": "user", "content": "你好！请用一句话介绍一下你自己。"},
    ],
)

# 第 5 步：把回复打印出来
# response.choices[0].message.content 就是 AI 回复的文字内容
print(response.choices[0].message.content)
