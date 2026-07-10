"""
08_agent_loop.py — 真正的 Agent 循环（ReAct 模式）
Agent 在一个循环里反复：思考 → 调工具 → 观察结果 → 再思考 → ... → 最终回答
"""

import os
import json
import math
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============ Embedding 搜索（用本地 Ollama） ============
ollama = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

KNOWLEDGE_LIST = [
    "贵阳是贵州省省会，气候凉爽，被称为'中国避暑之都'，夏季平均气温23°C。",
    "贵州省位于中国西南部，简称'黔'或'贵'，省会是贵阳。著名景点有黄果树瀑布、梵净山。",
    "贵州茅台是中国最著名的白酒品牌，产于贵州省遵义市茅台镇，有'国酒'之称。",
    "Python 由 Guido van Rossum 于 1991 年创造，是一种以简洁易读著称的高级编程语言。",
    "Transformer 架构于 2017 年由 Google 在论文 'Attention Is All You Need' 中提出。",
    "DeepSeek 是由中国深度求索公司开发的大语言模型，性价比极高，广泛应用于 AI Agent 开发。",
]

# 把知识库预计算成向量（程序启动时做一次，后面搜索不用重复算）
print("正在为知识库建立向量索引...")
kb_vectors = []
for doc in KNOWLEDGE_LIST:
    resp = ollama.embeddings.create(model="bge-embed", input=doc)
    kb_vectors.append(resp.data[0].embedding)
print(f"  共 {len(kb_vectors)} 条知识，向量索引已就绪\n")


def semantic_search(query, top_k=3):
    """用向量相似度搜索，而不是字符串匹配"""
    # 把查询转向量
    r = ollama.embeddings.create(model="bge-embed", input=query)
    q_vec = r.data[0].embedding

    # 跟每条知识算相似度
    scores = []
    for i, vec in enumerate(kb_vectors):
        dot = sum(a * b for a, b in zip(q_vec, vec))
        len_q = math.sqrt(sum(a * a for a in q_vec))
        len_k = math.sqrt(sum(a * a for a in vec))
        sim = dot / (len_q * len_k)
        scores.append((sim, i))

    scores.sort(reverse=True)
    results = []
    for sim, i in scores[:top_k]:
        results.append(f"(相似度 {sim:.4f}) {KNOWLEDGE_LIST[i]}")
    return "\n".join(results)


# ============ 工具定义 ============

# ① 工具菜单（给 AI 看的）
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库里搜索相关内容。适合查询事实、概念、人物。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词或问题"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。加减乘除、平方根、次方等。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '3*5+2'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

# ② 工具的实际实现
def search_knowledge(query):
    """用语义搜索在知识库里找相关内容（不再是字符串匹配）"""
    return semantic_search(query)


def calculate(expression):
    """安全计算数学表达式"""
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"


def get_current_time():
    return f"当前时间：{datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}"


TOOL_MAP = {
    "search_knowledge": search_knowledge,
    "calculate": calculate,
    "get_current_time": get_current_time,
}

# ============ Agent 核心循环 ============
MAX_STEPS = 5  # 最多调用 5 次工具，防止死循环


def run_agent(user_question, verbose=True):
    """
    ReAct 循环：
    while 还没结束:
        AI 思考 → 决定调工具还是直接回答
        如果调工具 → 执行 → 把结果加入对话 → 回到循环开头
        如果直接回答 → 结束循环，返回答案
    """
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，可以调用工具来获取信息。\n"
                "请一步一步思考：先用工具获取必要信息，再基于信息给用户完整准确的回答。\n"
                "如果第一次搜索没找到，可以换个关键词再搜。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    step = 0
    while step < MAX_STEPS:
        step += 1

        # ① 发送消息 + 工具列表给 AI
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        # ② 如果 AI 直接回答（不调工具）→ 结束
        if msg.tool_calls is None:
            if verbose:
                print(f"  [步骤{step}] AI 直接回答，不调工具")
            return msg.content, messages

        # ③ AI 要求调工具 → 执行
        messages.append(msg)  # 把 AI 的工具调用请求加入对话

        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  [步骤{step}] 调用工具：{name}({args})")

            func = TOOL_MAP[name]
            result = func(**args)

            if verbose:
                print(f"  [步骤{step}] 工具结果：{result[:80]}...")

            # 把工具结果加入对话（注意 role="tool"）
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result,
            })

    # 超出步数限制，强制让 AI 总结
    if verbose:
        print("  [警告] 达到最大步数限制，强制结束")
    messages.append({"role": "user", "content": "请基于已有信息给出回答。"})
    final = client.chat.completions.create(model="deepseek-chat", messages=messages)
    return final.choices[0].message.content, messages


# ============ 测试 ============
if __name__ == "__main__":
    tests = [
        "请介绍一下贵阳和贵州茅台",
        "帮我算一下 123 * 456 + 789，然后告诉我结果",
        "现在几点了？",
        "请告诉我 Python 是谁创造的，以及 Transformer 是哪年提出的",
    ]

    for q in tests:
        print("=" * 60)
        print(f"用户：{q}")
        print("-" * 60)
        answer, _ = run_agent(q, verbose=True)
        print(f"\n最终回答：{answer}\n")
