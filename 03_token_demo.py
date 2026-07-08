"""
03_token_demo.py — 理解 Token
功能：让你亲眼看到 AI 是怎么把文字切成 token 的
token 是 AI 计费的单位，也是它理解文字的最小单元
"""

import tiktoken

# cl100k_base 是 GPT-4 / DeepSeek 用的分词编码器
encoding = tiktoken.get_encoding("cl100k_base")

# ==================== 实验 1：中文的分词 ====================
text_cn = "你好，我想了解一下人工智能。"
tokens_cn = encoding.encode(text_cn)

print("实验 1：中文分词")
print(f"原文：{text_cn}")
print(f"词元数量：{len(tokens_cn)} 个 token")
for i, token_id in enumerate(tokens_cn):
    token_text = encoding.decode([token_id])
    print(f"  token[{i}] = {token_id} -> \"{token_text}\"")
print()

# ==================== 实验 2：英文的分词 ====================
text_en = "Hello, I want to learn about artificial intelligence."
tokens_en = encoding.encode(text_en)

print("实验 2：英文分词")
print(f"原文：{text_en}")
print(f"词元数量：{len(tokens_en)} 个 token")
for i, token_id in enumerate(tokens_en):
    token_text = encoding.decode([token_id])
    print(f"  token[{i}] = {token_id} -> \"{token_text}\"")
print()

# ==================== 实验 3：为什么 token 很重要 ====================
print("实验 3：多轮对话的 token 累积")
print("-" * 40)

# 模拟一段对话历史
conversation = [
    ("system", "你是一个友好的助手。"),
    ("user", "帮我写一篇关于人工智能的文章。"),
    ("assistant", "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，它研究如何让计算机模拟人类智能行为..."),
    ("user", "再详细一点。"),
    ("assistant", "好的，让我从几个方面详细展开。首先是机器学习，它是AI的核心方法之一，通过从数据中学习规律来做出预测..."),
]

total = 0
for role, content in conversation:
    count = len(encoding.encode(content))
    total += count
    print(f"[{role}] {content[:30]}... -> {count} token")

print(f"\n这段对话历史总共约 {total} 个 token")
print()
print("要点：")
print("- 中文 1-2 个字符 ≈ 1 个 token，英文 3-4 个字母 ≈ 1 个 token")
print("- token 越多 → 费用越高 → 速度越慢")
print("- DeepSeek 一次最多约 128k token，超出就截断最前面的")
print("- Agent 来回调用多次，token 会越堆越多，必须学会管理")
