# 学习笔记 — Day 1（2026.07.08）

## 今天学的库

### 1. `openai` — 调大模型 API
```python
from openai import OpenAI
```
| 用法 | 作用 | 在哪见过 |
|------|------|---------|
| `OpenAI(api_key=..., base_url=...)` | 创建客户端，连上大模型服务器 | 所有脚本 |
| `client.chat.completions.create(model=..., messages=...)` | 发送消息给 AI，等它回复 | 所有脚本 |
| `response.choices[0].message.content` | 从返回值里取出 AI 回复的**文字** | 所有脚本 |
| `response.usage.prompt_tokens` | 输入花了多少 token | 03 |
| `response.usage.completion_tokens` | 输出花了多少 token | 03 |
| `response.usage.total_tokens` | 总共花了多少 token | 03 |

### 2. `python-dotenv` — 读 `.env` 文件
```python
from dotenv import load_dotenv
```
| 用法 | 作用 |
|------|------|
| `load_dotenv()` | 把 `.env` 文件里的内容加载成环境变量 |

### 3. `os` — 操作系统的功能（Python 自带）
```python
import os
```
| 用法 | 作用 |
|------|------|
| `os.getenv("变量名")` | 读取环境变量的值 |

### 4. `tiktoken` — 文字切 token
```python
import tiktoken
```
| 用法 | 作用 |
|------|------|
| `tiktoken.get_encoding("cl100k_base")` | 获取分词器 |
| `encoding.encode("文字")` | 把文字切成 token，返回 token ID 列表 |
| `encoding.decode([token_id])` | 把 token ID 转回文字 |

---

## 今天学的核心概念

### messages 结构
```python
messages = [
    {"role": "system",    "content": "设定 AI 角色的指令"},    # 只有开发者能写
    {"role": "user",      "content": "用户说的话"},
    {"role": "assistant", "content": "AI 说的话（回复）"},
]
```
- **AI 自己没有记忆**。每次都要把完整的 messages 列表重新发送
- 对话越长 → messages 越重 → token 越多 → 越贵越慢

### token
- AI 不读字，读 token（一段一段的数字 ID）
- 中文 1-2 个字 ≈ 1 token，英文 1 个单词 ≈ 1-2 token
- DeepSeek 上限约 128k token（超出就截断）

### `.env` 文件
- 存密码和配置的地方
- 永远不提交到 GitHub（`.gitignore` 里有它）
- `load_dotenv()` 读取 → `os.getenv()` 取值

---

### `openai` 补充 — Embedding 也可以用这个库
| 用法 | 作用 |
|------|------|
| `client.embeddings.create(model=..., input=...)` | 把文字转成向量 |
| `resp.data[0].embedding` | 取出那串向量（1024 个小数） |

### 5. Cosine Similarity（余弦相似度）— 自写的
```python
def cosine_similarity(vec_a, vec_b):
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    len_a = math.sqrt(sum(a * a for a in vec_a))
    len_b = math.sqrt(sum(b * b for b in vec_b))
    return dot / (len_a * len_b)
```
两个向量夹角越小 → 越接近 1 → 意思越相近

---

## 今天写的脚本

| 文件 | 干什么 |
|------|--------|
| `01_hello_llm.py` | 最简单的 API 调用，问一句答一句 |
| `02_chat_loop.py` | 多轮对话循环，理解 messages 累积 |
| `03_token_demo.py` | 看 AI 怎么切 token，理解 token 消耗 |
| `04_embedding_demo.py` | 把文字变向量，比较意思相似度 |
| `05_rag_simple.py` | **第一个 RAG 系统**：检索 + 增强生成 |

---

## RAG 流程（核心中的核心）

```
用户问题 → Embedding 变向量 → 跟知识库每条向量比相似度
→ 取最相似的 2-3 条 → 拼进 prompt → 发给 LLM → LLM 基于真实信息回答
```

RAG = Retrieval（检索）+ Augmented（增强）+ Generation（生成）
