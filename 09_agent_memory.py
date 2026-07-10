"""
09_agent_memory.py — Agent Memory 管理
解决的问题：对话无限增长 → token 爆炸 → 需要自动裁剪
方案：设定 token 上限，超出时保留 system prompt + 最近 N 条消息
"""

import os
import json
import math
import tiktoken
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
ollama = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# ============ Token 管理 ============
encoder = tiktoken.get_encoding("cl100k_base")
MAX_TOKENS = 8000  # 设大一些，让多轮对话能正常运转


def to_dict(msg):
    """把 OpenAI 返回的 Pydantic 对象统一转成 dict，方便后续处理"""
    if isinstance(msg, dict):
        return msg
    # ChatCompletionMessage → dict
    d = {"role": msg.role}
    if msg.content:
        d["content"] = msg.content
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in msg.tool_calls
        ]
    if hasattr(msg, "tool_call_id") and msg.tool_call_id:
        d["tool_call_id"] = msg.tool_call_id
    return d


def count_tokens(messages):
    total = 0
    for msg in messages:
        content = msg.get("content", "") or ""
        total += len(encoder.encode(content))
    return total


def trim_messages(messages, max_tokens=MAX_TOKENS):
    """
    裁剪对话：从最旧的消息开始丢弃，但保证 tool_calls + tool 响应成对丢弃。
    思路：找到每条 user 消息的起始位置，按"轮"丢弃。每轮 = user + 后续所有直到下一条 user。
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    # 把 other_msgs 按"轮"分组（每轮从 user 消息开始）
    rounds = []
    current_round = []
    for msg in other_msgs:
        if msg.get("role") == "user" and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)
    if current_round:
        rounds.append(current_round)

    # 从最新轮往前保留，直到 token 超标
    kept_rounds = []
    kept_tokens = sum(len(encoder.encode(m.get("content", "") or "")) for m in system_msgs)

    for r in reversed(rounds):
        r_tokens = sum(len(encoder.encode(m.get("content", "") or "")) for m in r)
        if kept_tokens + r_tokens <= max_tokens:
            kept_rounds.insert(0, r)
            kept_tokens += r_tokens
        else:
            break  # 这轮放不下，更早的轮也不放了

    # 展开成扁平列表
    kept = []
    for r in kept_rounds:
        kept.extend(r)

    trimmed = system_msgs + kept
    dropped = len(other_msgs) - len(kept)
    if dropped > 0:
        trimmed.insert(len(system_msgs), {"role": "system", "content": f"[已丢弃 {dropped} 条旧对话]"})
    return trimmed


# ============ 知识库和工具（跟 08 一样，略） ============
KNOWLEDGE_LIST = [
    "贵阳是贵州省省会，气候凉爽，被称为'中国避暑之都'，夏季平均气温23°C。",
    "贵州茅台是中国最著名的白酒品牌，产于贵州省遵义市茅台镇，有'国酒'之称。",
    "Python 由 Guido van Rossum 于 1991 年创造。",
    "Transformer 架构于 2017 年由 Google 提出。",
]

print("正在建立知识库向量索引...")
kb_vectors = []
for doc in KNOWLEDGE_LIST:
    resp = ollama.embeddings.create(model="bge-embed", input=doc)
    kb_vectors.append(resp.data[0].embedding)
print("就绪\n")


def semantic_search(query, top_k=2):
    r = ollama.embeddings.create(model="bge-embed", input=query)
    q_vec = r.data[0].embedding
    scores = []
    for i, vec in enumerate(kb_vectors):
        dot = sum(a * b for a, b in zip(q_vec, vec))
        len_q = math.sqrt(sum(a * a for a in q_vec))
        len_k = math.sqrt(sum(a * a for a in vec))
        scores.append((dot / (len_q * len_k), i))
    scores.sort(reverse=True)
    return "\n".join(
        f"{KNOWLEDGE_LIST[i]}" for _, i in scores[:top_k]
    )


def calculate(expression):
    try:
        return f"{expression} = {eval(expression, {'__builtins__': {}}, math.__dict__)}"
    except Exception as e:
        return f"出错：{e}"


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库里搜索相关内容",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "搜索关键词"}},
                "required": ["query"],
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
                "properties": {"expression": {"type": "string", "description": "数学表达式"}},
                "required": ["expression"],
            },
        },
    },
]

TOOL_MAP = {"search_knowledge": semantic_search, "calculate": calculate}

# ============ Agent（带 Memory 管理） ============
MAX_STEPS = 5


def run_agent(user_question, messages=None, verbose=True):
    """
    带 Memory 的 Agent：
    - messages 可以跨轮复用（多轮对话）
    - 每轮自动检查 token 数，超出就裁剪
    """
    if messages is None:
        messages = [
            {"role": "system", "content": "你是一个智能助手，可以调用工具获取信息，用中文回答。"},
        ]

    messages.append({"role": "user", "content": user_question})

    step = 0
    while step < MAX_STEPS:
        step += 1

        # ★ 每次发请求前，检查并裁剪 token
        token_count = count_tokens(messages)
        if verbose:
            print(f"  [Memory] 当前 token 数：{token_count}")
        if token_count > MAX_TOKENS:
            messages = trim_messages(messages, MAX_TOKENS)
            if verbose:
                print(f"  [Memory] 裁剪后 token 数：{count_tokens(messages)}")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = response.choices[0].message

        if msg.tool_calls is None:
            messages.append(to_dict(msg))
            return msg.content, messages

        messages.append(to_dict(msg))
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            result = TOOL_MAP[name](**args)
            if verbose:
                print(f"  [步骤{step}] {name}({args}) → {result[:50]}...")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    messages.append({"role": "user", "content": "请基于已有信息回答。"})
    final = client.chat.completions.create(model="deepseek-chat", messages=messages)
    messages.append(to_dict(final.choices[0].message))
    return final.choices[0].message.content, messages


# ============ 演示：多轮对话 + Memory 裁剪 ============
if __name__ == "__main__":
    print("=" * 60)
    print("演示：多轮对话 + Memory 自动裁剪")
    print("=" * 60)

    # 用一个 messages 变量跨多轮复用
    conversation = [
        {"role": "system", "content": "你是一个智能助手，用中文回答，回答尽量详细。"},
    ]

    questions = [
        "请用一段比较长的文字介绍一下贵阳这座城市",  # 会得到长回答
        "那贵州茅台呢？",                              # 指代，需要上下文
        "帮我算 999 * 888",                           # 调计算器
        "前面说的那两个城市分别是什么省的省会？",        # 跨轮指代
    ]

    for i, q in enumerate(questions):
        print(f"\n--- 第 {i+1} 轮 ---")
        print(f"用户：{q}")
        answer, conversation = run_agent(q, messages=conversation)
        print(f"AI：{answer[:150]}...")
        print(f"  会话历史共 {len(conversation)} 条消息")
