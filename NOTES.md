# 学习笔记 — Week 1（2026.07.08）

---

## 一、架构全景图

从零开始，7 个脚本的进化路线：

```
01_hello_llm       最简单的 API 调用，问一句答一句
    │
    ▼
02_chat_loop       多轮对话，理解 messages 列表就是"记忆"
    │
    ▼
03_token_demo      理解 token：AI 不读字，读数字 ID
    │
    ▼
04_embedding_demo  把文字变成 1024 个小数的向量，意思相近=向量相近
    │
    ▼
05_rag_simple      检索+生成：先搜知识库，再喂给 LLM，不瞎编
    │
    ▼
06_chromadb_rag    向量存硬盘（ChromaDB），程序关了也不丢
    │
    ▼
07_function_calling  AI 学会"调工具"——Agent 的基石
                     AI 说"我要调 XX 工具，参数是 YY"
                     → 你的代码执行 → 结果返回 AI → AI 继续回答
```

**整条路线的核心思想**：让 AI 从"只能聊天"变成"能查知识库、能调工具、能做事"。

---

## 二、术语中英对照表

| 英文 | 中文 | 一句话解释 |
|------|------|-----------|
| LLM (Large Language Model) | 大语言模型 | GPT、DeepSeek、Claude 这类 AI |
| Token | 词元 / 令牌 | AI 读文字的最小单元，不是字 |
| Embedding / Vector | 向量嵌入 / 向量 | 把文字变成一串小数（1024 个），方便数学计算 |
| Cosine Similarity | 余弦相似度 | 算两个向量有多像，-1~1，越接近 1 越像 |
| RAG (Retrieval-Augmented Generation) | 检索增强生成 | 先搜知识库 → 再让 LLM 基于搜到的内容回答 |
| Vector Database | 向量数据库 | 专门存向量 + 快速搜相似向量的数据库（如 ChromaDB） |
| Prompt | 提示词 | 你发给 AI 的那段话 |
| System Prompt | 系统提示词 | `role: "system"` 那条，给 AI 定角色、定规矩 |
| Function Calling / Tool Use | 函数调用 / 工具调用 | AI 决定调哪个工具、传什么参数，你的代码负责执行 |
| Agent | 智能体 | 能自主规划、调用工具、多步执行的 AI 系统 |
| ReAct | 推理-行动循环 | Thought → Action → Observation → Thought... 循环 |
| MCP (Model Context Protocol) | 模型上下文协议 | 让 AI 连接外部工具的标准化协议 |
| API Key | API 密钥 | 你的身份证，证明你有权限调大模型 |

---

## 三、学到的库和函数

### 1. `openai` — 调大模型 API（最核心的库）

```python
from openai import OpenAI
```

#### 创建客户端
| 用法 | 作用 |
|------|------|
| `OpenAI(api_key=..., base_url=...)` | 创建客户端。`api_key` 是身份凭证，`base_url` 是服务器地址（DeepSeek / Ollama 不同） |

#### 对话（Chat Completions）
| 用法 | 作用 |
|------|------|
| `client.chat.completions.create(model=..., messages=...)` | 发送消息给 AI |
| `client.chat.completions.create(model=..., messages=..., tools=...)` | 发送消息 + 工具列表（Function Calling） |

#### 从对话返回值里取数据（属性链）
| 属性 | 类型 | 作用 |
|------|------|------|
| `response.choices` | list | 候选回复列表，通常只取 `[0]` |
| `response.choices[0].message` | object | AI 的回复消息 |
| `response.choices[0].message.content` | str 或 None | AI 回复的**文字内容**。如果是工具调用则为 None |
| `response.choices[0].message.tool_calls` | list 或 None | AI 要调的工具列表。不调工具时是 None |
| `response.choices[0].message.tool_calls[0].id` | str | 工具调用的唯一 ID |
| `response.choices[0].message.tool_calls[0].function.name` | str | AI 要调的工具名 |
| `response.choices[0].message.tool_calls[0].function.arguments` | str (JSON) | AI 传的参数（JSON 字符串，需 `json.loads()` 解析） |
| `response.usage.prompt_tokens` | int | 输入消耗的 token 数 |
| `response.usage.completion_tokens` | int | 输出消耗的 token 数 |
| `response.usage.total_tokens` | int | 总共消耗的 token 数 |

#### Embedding（向量嵌入）
| 用法 | 作用 |
|------|------|
| `client.embeddings.create(model=..., input=...)` | 把文字转成向量 |
| `resp.data[0].embedding` | 取出向量（list of float，如 1024 个小数） |

#### messages 的 4 种 role
| role | 谁说的 | 什么时候用 |
|------|--------|-----------|
| `"system"` | 开发者设定 | 给 AI 定角色、定规则 |
| `"user"` | 用户 | 用户的提问 |
| `"assistant"` | AI | AI 的回复 |
| `"tool"` | 工具执行结果 | **必须有 `tool_call_id`**，告诉 AI 这是哪个工具调用的结果 |

---

### 2. `python-dotenv` — 读 `.env` 文件

```python
from dotenv import load_dotenv
```

| 函数 | 作用 |
|------|------|
| `load_dotenv()` | 执行后，`.env` 里的内容变成环境变量 |

---

### 3. `os` — Python 自带，操作系统相关

```python
import os
```

| 函数 | 作用 |
|------|------|
| `os.getenv("变量名")` | 读取环境变量，`.env` 里写了啥就能读到啥 |

---

### 4. `tiktoken` — 文字切 token

```python
import tiktoken
```

| 函数/属性 | 作用 |
|-----------|------|
| `tiktoken.get_encoding("cl100k_base")` | 获取分词编码器 |
| `encoding.encode("文字")` | 把文字切成 token，返回整数列表 |
| `encoding.decode([token_id])` | 把 token ID 列表还原成文字 |
| `len(tokens)` | token 数量 |

---

### 5. `chromadb` — 向量数据库

```python
import chromadb
```

| 函数/属性 | 作用 |
|-----------|------|
| `chromadb.PersistentClient(path="./chroma_db")` | 创建客户端，数据存在 `path` 文件夹 |
| `client.get_or_create_collection(name="表名")` | 获取或创建一张"表"（collection） |
| `collection.count()` | 表里有多少条记录 |
| `collection.add(ids=[...], documents=[...], embeddings=[...])` | 添加记录（ID、原文、向量） |
| `collection.query(query_embeddings=[...], n_results=2, include=["documents", "distances"])` | 按向量相似度搜索，返回最相似的 `n_results` 条 |
| `results["documents"][0]` | 搜索结果里的原文列表 |
| `results["distances"][0]` | 搜索结果里的距离列表（越小越相似） |

---

### 6. `math` — Python 自带，数学运算

```python
import math
```

| 函数 | 作用 |
|------|------|
| `math.sqrt(x)` | 开平方根（余弦相似度公式里用来算向量长度） |

---

### 7. `json` — Python 自带，处理 JSON

```python
import json
```

| 函数 | 作用 |
|------|------|
| `json.loads(json_string)` | 把 JSON 字符串转成 Python dict（AI 返回的 tool arguments 是 JSON 格式，需要解析） |

---

### 8. 自建函数汇总

| 函数 | 在哪 | 作用 |
|------|------|------|
| `cosine_similarity(vec_a, vec_b)` | 04, 05 | 算两个向量的余弦相似度 |
| `get_embedding(text)` | 05, 06 | 把任意文字转成向量（封装 `ollama.embeddings.create`） |
| `retrieve(question, top_k)` | 05, 06 | 根据问题在知识库里检索最相关的段落 |
| `answer(question, docs)` | 05, 06 | 把检索结果拼进 prompt，用 DeepSeek 生成回答 |
| `get_weather(city)` | 07 | 模拟天气查询（真实项目会对接天气 API） |
| `calculate(expression)` | 07 | 安全计算数学表达式 |
| `run_agent(user_question)` | 07 | 核心 Agent 循环：发消息→AI 决定调工具→执行→返回结果→继续 |

---

### 9. Python 内置函数/语法（今天新碰到的）

| 用法 | 作用 | 在哪见过 |
|------|------|---------|
| `input("提示文字")` | 程序暂停，等用户在终端打字 | 02, 06 |
| `print(f"文字 {变量}")` | f-string，花括号里放变量，打印时自动替换 | 所有脚本 |
| `while True:` | 无限循环 | 02, 06 |
| `break` | 跳出当前循环 | 02, 06 |
| `list.append(元素)` | 在列表末尾追加元素 | 02 |
| `str.strip()` | 去掉字符串头尾的空格 | 02, 06 |
| `str.lower()` | 全部转小写字母 | 02, 06 |
| `str.startswith("前缀")` | 判断字符串是否以某个前缀开头 | 06 |
| `sum(...)` | 求和 | 04, 05 |
| `zip(list_a, list_b)` | 把两个列表配对成元组 | 04, 05 |
| `round(x, 4)` | 四舍五入保留 4 位小数 | 06 |
| `enumerate(list)` | 同时遍历索引和值 | 03, 04, 06 |
| `"字符串" * 50` | 字符串重复 50 次（画分割线） | 02 |
| `eval("表达式")` | 执行字符串里的 Python 表达式（有安全风险！） | 07 |
| `func(**dict)` | 把字典展开成关键字参数传给函数 | 07 |
| `list.sort(reverse=True)` | 列表降序排序 | 05 |
| `try: ... except Exception:` | 捕获异常，防止程序崩溃 | 06, 07 |
| `json.loads(str)` | JSON 字符串 → Python 对象 | 07 |

---

## 四、7 个脚本对应关系

| 脚本 | 学到的核心概念 | 新增的库 |
|------|--------------|---------|
| `01_hello_llm.py` | API 调用、messages 基础结构 | `openai`, `dotenv`, `os` |
| `02_chat_loop.py` | 多轮对话、messages 累积就是"记忆" | — |
| `03_token_demo.py` | Token 切分、为什么 token 很重要 | `tiktoken` |
| `04_embedding_demo.py` | 向量嵌入、余弦相似度、意思相近=向量相近 | `math` |
| `05_rag_simple.py` | RAG 完整流程：检索→增强→生成 | — |
| `06_chromadb_rag.py` | 向量数据库持久化、抽函数消除重复代码 | `chromadb` |
| `07_function_calling.py` | 工具调用、Agent 的核心机制 | `json` |

---

## 五、下次要学

Agent 进阶：Memory 管理 + ReAct 循环 + 多工具自动路由
