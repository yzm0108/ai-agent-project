"""
11_agentic_rag.py — Agentic RAG
普通 RAG：用户问 → 检索 → 回答（死活只搜一次）
Agentic RAG：Agent 决定 → 要不要搜？搜什么词？搜到没有？不够再搜！
"""

import os
import json
import math
import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)
ollama = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

# ============ 知识库（ChromaDB + 语义搜索） ============
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="agentic_kb")

if collection.count() == 0:
    docs = [
        "贵阳是贵州省省会，气候凉爽，夏季平均气温23°C，被称为'中国避暑之都'。著名景点包括甲秀楼、黔灵山公园、青岩古镇。",
        "贵州茅台产于贵州省遵义市茅台镇，是中国最著名的酱香型白酒，被誉为'国酒'，采用传统12987工艺酿造。",
        "Python 由 Guido van Rossum 于1991年创造，以简洁、易读著称。常用来开发Web应用、数据分析、AI系统。",
        "Docker 是一个容器化平台，于2013年开源。它让开发者把应用和依赖打包成一个容器，到处运行。",
        "Transformer 架构在2017年由 Google 团队提出，论文标题是 'Attention Is All You Need'。它是 GPT、BERT 等模型的基础。",
        "RAG 是检索增强生成（Retrieval-Augmented Generation）的缩写，2020年由 Facebook AI 提出。先检索再生成，减少幻觉。",
        "Agentic RAG 是在普通 RAG 上加入 Agent 的决策能力。Agent 自主决定检索策略、改写查询、判断信息是否足够。",
        "FAISS 是 Facebook 开源的向量相似度搜索库。ChromaDB 是轻量级向量数据库。Milvus 是云原生分布式向量数据库。",
    ]
    for i, doc in enumerate(docs):
        emb = ollama.embeddings.create(model="bge-embed", input=doc)
        collection.add(ids=[str(i)], documents=[doc], embeddings=[emb.data[0].embedding])
    print(f"知识库初始化完成，共 {collection.count()} 条\n")


def search_knowledge(query, top_k=3):
    """语义搜索知识库"""
    q_emb = ollama.embeddings.create(model="bge-embed", input=query)
    results = collection.query(
        query_embeddings=[q_emb.data[0].embedding],
        n_results=top_k,
        include=["documents", "distances"],
    )
    lines = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        lines.append(f"(相似度 {1-dist:.4f}) {doc}")
    return "\n".join(lines)


# ============ Agent 工具定义 ============
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "在知识库中搜索信息。注意：\n"
                "- 把问题拆成关键词再搜，不要整句话丢进去\n"
                "- 如果第一次搜索结果不够，换个关键词再搜\n"
                "- 复杂问题可以分多次搜索不同方面"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词（用空格分隔多个词）"},
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {"search_knowledge": search_knowledge}

# ============ Agentic RAG 核心 ============
SYSTEM_PROMPT = """你是一个 Agentic RAG 助手。知识库中有一些文档，你可以通过搜索来获取信息。

关键原则：
1. 收到问题后，先思考"需不需要搜索？"简单闲聊不搜，事实性问题一定要搜
2. 搜索时把问题拆成关键词（如"贵阳 景点 美食"），不要丢整句话
3. 如果搜索结果不相关或不够，换个角度、换个关键词再搜
4. 搜索够了就整合信息，给出完整的回答
5. 如果多次搜索都找不到，诚实告诉用户"知识库中没有相关信息"

请严格基于搜索结果回答。搜索结果里没有的，不要编造。"""


def run_agentic_rag(question, verbose=True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    step = 0
    while step < 8:
        step += 1

        resp = deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=TOOLS,
        )
        msg = resp.choices[0].message

        # AI 决定直接回答
        if msg.tool_calls is None:
            if verbose:
                print(f"  [步骤{step}] AI 直接回答（不调工具）")
            return msg.content

        # AI 要搜索
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            if verbose:
                print(f"  [步骤{step}] 搜索：{args['query']}")

            result = TOOL_MAP[tc.function.name](**args)

            if verbose:
                # 展示搜到了什么
                for line in result.split("\n")[:2]:
                    print(f"          → {line[:80]}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    # 到头了，强制总结
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=messages + [{"role": "user", "content": "请基于搜索到的信息给出最终回答。"}],
    )
    return resp.choices[0].message.content


# ============ 测试：对比普通 RAG vs Agentic RAG ============
if __name__ == "__main__":
    print("=" * 60)
    print("测试 1：简单事实查询")
    print("=" * 60)
    q1 = "贵阳有什么好玩的？"
    print(f"用户：{q1}")
    ans = run_agentic_rag(q1)
    print(f"\nAI：{ans}\n")

    print("=" * 60)
    print("测试 2：需要多次搜索才能拼出完整答案")
    print("=" * 60)
    q2 = "Python 和 Docker 分别是什么时候诞生的？各有什么特点？"
    print(f"用户：{q2}")
    ans = run_agentic_rag(q2)
    print(f"\nAI：{ans}\n")

    print("=" * 60)
    print("测试 3：知识库里没有的")
    print("=" * 60)
    q3 = "马斯克是什么时候收购推特的？"
    print(f"用户：{q3}")
    ans = run_agentic_rag(q3)
    print(f"\nAI：{ans}\n")

    print("=" * 60)
    print("测试 4：Agent 自主判断不需要搜索")
    print("=" * 60)
    q4 = "你好！请用一句话介绍你自己。"
    print(f"用户：{q4}")
    ans = run_agentic_rag(q4)
    print(f"\nAI：{ans}")
