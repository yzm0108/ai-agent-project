"""
04_embedding_demo.py — 理解 Embedding（向量嵌入）
用本地 Ollama 的 embedding 模型，不花钱、不联网
"""

from openai import OpenAI

# 连接本地 Ollama（确保 Ollama 正在运行）
client = OpenAI(
    api_key="ollama",                          # Ollama 本地不需要真 key，随便写
    base_url="http://localhost:11434/v1",      # Ollama 的 OpenAI 兼容接口
)

# ==================== 准备 4 段文字 ====================
texts = {
    "① 关于猫": "猫是一种常见的宠物，它们性格独立，喜欢睡觉和追逐小动物。",
    "② 还是猫": "小猫非常可爱，它们会用爪子洗脸，喜欢玩毛线球。",
    "③ 关于狗": "狗是人类忠实的伙伴，它们需要每天遛弯，喜欢跟主人互动。",
    "④ 电脑":   "Python 是一门高级编程语言，广泛用于数据科学和人工智能领域。",
}


def get_embedding(text):
    """把一段文字转成向量（一串小数）"""
    resp = client.embeddings.create(
        model="bge-embed",             # BGE 中文 embedding，效果好
        input=text,
    )
    return resp.data[0].embedding


print("正在为每段文字生成向量（使用本地 Ollama）...")
embeddings = {}
for name, text in texts.items():
    emb = get_embedding(text)
    embeddings[name] = emb
    print(f"  {name} 完成，向量长度 = {len(emb)}")

print()


# ==================== 计算相似度 ====================
import math


def cosine_similarity(vec_a, vec_b):
    """余弦相似度：-1 到 1，越接近 1 说明意思越相近"""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    len_a = math.sqrt(sum(a * a for a in vec_a))
    len_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (len_a * len_b)


print("两两比较相似度：")
print("-" * 50)
names = list(embeddings.keys())
for i, name_a in enumerate(names):
    for name_b in names[i + 1:]:
        sim = cosine_similarity(embeddings[name_a], embeddings[name_b])
        bar = "█" * max(1, int((sim + 0.3) * 10))
        print(f"  {name_a} vs {name_b}: {sim:.4f} {bar}")

print()
print("解读：")
print("  ① vs ②（都是猫）→ 相似度应该最高")
print("  ① vs ③（猫 vs 狗）→ 中等，都是宠物")
print("  ① vs ④（猫 vs 编程）→ 最低，完全无关")
print()
print("这就是 RAG 的核心原理：用户问题 → 向量 → 找最相似的段落 → 喂给 LLM")
