"""
05_rag_simple.py — 第一个 RAG 系统
功能：一段一段找跟问题最相关的文本，让 LLM 基于它回答
RAG = Retrieval（检索）+ Augmented（增强）+ Generation（生成）
"""

import os
import math
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============ 两个客户端：一个调 Ollama（本地 embedding），一个调 DeepSeek（回答） ============
ollama = OpenAI(
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)

deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============ 知识库：模拟 6 篇小文章 ============
knowledge_base = [
    "贵阳是贵州省的省会，位于中国西南部，气候凉爽，被称为'中国避暑之都'。",
    "贵州茅台是中国最著名的白酒品牌之一，原产地在贵州省遵义市的茅台镇。",
    "Python 是由 Guido van Rossum 于 1991 年创造的编程语言，以简洁易读著称。",
    "大语言模型（LLM）是通过海量文本训练的神经网络，代表包括 GPT、Claude 和 DeepSeek。",
    "黄果树瀑布位于贵州省安顺市，是亚洲最大的瀑布之一，高约 77.8 米。",
    "Transformer 架构于 2017 年由 Google 提出，是现代大语言模型的基础结构。",
]


# ============ 辅助函数 ============
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    len_a = math.sqrt(sum(a * a for a in vec_a))
    len_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (len_a * len_b)


def get_embedding(text):
    resp = ollama.embeddings.create(model="bge-embed", input=text)
    return resp.data[0].embedding


# ============ 第 1 步：提前把所有知识转成向量（只做一次） ============
print("正在建立知识库向量索引...")
kb_embeddings = []  # 存每条知识的向量
for i, doc in enumerate(knowledge_base):
    emb = get_embedding(doc)
    kb_embeddings.append(emb)
    print(f"  [{i}] {doc[:40]}... -> 向量完成")
print(f"知识库共 {len(knowledge_base)} 条，已全部向量化\n")


# ============ 第 2 步：检索 —— 找到跟问题最相似的 2 条知识 ============
def retrieve(question, top_k=2):
    """输入问题，返回知识库里最相关的 top_k 条"""
    q_emb = get_embedding(question)  # ① 把问题也转成向量
    scores = []
    for i, kb_emb in enumerate(kb_embeddings):
        sim = cosine_similarity(q_emb, kb_emb)  # ② 跟每条知识比相似度
        scores.append((sim, i))
    scores.sort(reverse=True)  # ③ 按相似度降序排列
    results = []
    for sim, i in scores[:top_k]:  # ④ 取最相似的 top_k 条
        results.append((sim, knowledge_base[i]))
    return results


# ============ 第 3 步：回答 —— 把检索结果拼进 prompt，让 LLM 基于它回答 ============
def answer(question, retrieved_docs):
    """把检索到的相关段落拼进 system prompt，让 LLM 只能基于它们回答"""
    context = "\n".join([f"- {doc}" for _, doc in retrieved_docs])

    response = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一个基于知识库回答问题的助手。\n"
                    "以下是知识库中与用户问题相关的段落：\n"
                    f"{context}\n\n"
                    "请严格基于以上段落回答问题。如果段落中没有相关信息，请直接说'知识库中没有相关信息'。\n"
                    "不要编造任何不在段落中的内容。"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return response.choices[0].message.content


# ============ 测试：问 3 个问题 ============
questions = [
    "贵阳有什么特点？",
    "Python 是谁创造的？",
    "苹果公司的总部在哪里？",  # 知识库里没有，看看它会不会瞎编
]

for q in questions:
    print("=" * 60)
    print(f"问题：{q}")

    # 检索
    docs = retrieve(q)
    print("检索到的相关段落：")
    for sim, doc in docs:
        print(f"  (相似度 {sim:.4f}) {doc}")

    # 生成回答
    ans = answer(q, docs)
    print(f"\nAI 回答：{ans}")
    print()
