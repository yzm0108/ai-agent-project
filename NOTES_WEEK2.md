# 学习笔记 — Week 2（Agent 核心：Tool Use + Memory + ReAct）

---

## 一、本周架构全景图

```
07_function_calling     AI 学会"调工具"——Agent 的基石
    │                    工具菜单（tools）→ AI 决定调哪个 → 代码执行 → 返回结果
    │
    ▼
08_agent_loop          ReAct 循环：Think → Act → Observe → Think → ...
    │                    多步推理，调完一个工具看结果，不够再调下一个
    │                    语义搜索替换字符串匹配，查得更准
    │
    ▼
09_agent_memory        Memory 管理：对话太长 → token 超标 → 自动裁剪
                         按"轮"裁剪（user 消息为边界），保证 tool_calls+tool 成对
```

**核心思想**：让 AI 从"被动回答"变成"主动规划、调用工具、管理记忆"。

---

## 二、术语中英对照表（Week 2 新增）

| 英文 | 中文 | 一句话解释 |
|------|------|-----------|
| Function Calling | 函数调用 | LLM 说"我要调 XX 函数，参数是 YY"，你的代码负责执行 |
| Tool / Tool Definition | 工具 / 工具定义 | 给 AI 看的"菜单"，描述工具名、用途、参数 |
| ReAct | 推理-行动循环 | Reasoning + Acting，AI 反复：思考→行动→观察→再思考 |
| Tool Call | 工具调用请求 | AI 返回的不是文字，而是一个"请执行 XX 工具"的指令 |
| Semantic Search | 语义搜索 | 用 embedding 向量相似度搜索，而不是字符串匹配 |
| Token Limit | Token 上限 | LLM 一次能处理的 token 上限（DeepSeek 约 128k） |
| Memory Trimming | 记忆裁剪 | 对话太长时自动丢弃旧消息，保持 token 不超标 |
| Round（对话轮次） | 一轮对话 | 一条 user 消息 + 后续所有 assistant/tool 消息，直到下一条 user |

---

## 三、新学到的库和函数

### 1. `json` — Python 自带，处理 JSON

```python
import json
```

| 函数 | 作用 | 在哪见过 |
|------|------|---------|
| `json.loads(json_string)` | JSON 字符串 → Python dict | 07, 08, 09, 11 |
| `json.dumps(obj, ensure_ascii=False)` | Python 对象 → JSON 字符串 | 10, 11 |

**为什么需要**：AI 返回的 `tool_call.function.arguments` 是 JSON 字符串（如 `'{"city": "贵阳"}'`），必须用 `json.loads()` 转成 dict 才能取参数值。

### 2. `datetime` — Python 自带，日期时间

```python
from datetime import datetime
```

| 函数 | 作用 |
|------|------|
| `datetime.now()` | 获取当前日期时间对象 |
| `datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")` | 格式化成中文时间字符串 |

### 3. `math` — Python 自带（补充）

```python
import math
```

| 函数 | 作用 |
|------|------|
| `math.sqrt(x)` | 平方根（余弦相似度公式用） |
| `math.__dict__` | math 模块的所有函数字典（传给 `eval` 做安全沙箱） |

### 4. `tiktoken` 补充 — Token 计数用于 Memory 管理

```python
import tiktoken
encoder = tiktoken.get_encoding("cl100k_base")
```

| 用法 | 作用 |
|------|------|
| `len(encoder.encode("文字"))` | 计算一段文字占多少 token |

---

## 四、新学到的 OpenAI API 属性

### Function Calling 相关（07 开始用到）

| 属性 | 类型 | 作用 |
|------|------|------|
| `client.chat.completions.create(..., tools=TOOLS)` | — | 发送消息时附带工具列表 |
| `response.choices[0].message.tool_calls` | list 或 None | AI 要调用的工具列表。不调工具时是 `None` |
| `tool_call.id` | str | 本次工具调用的唯一 ID（用于 tool 消息配对） |
| `tool_call.function.name` | str | AI 要调的工具名 |
| `tool_call.function.arguments` | str (JSON) | AI 传的参数，需 `json.loads()` 解析 |

### messages 新增第 4 种 role

| role | 谁发的 | 什么时候用 |
|------|--------|-----------|
| `"system"` | 开发者 | 设定 AI 角色 |
| `"user"` | 用户 | 提问 |
| `"assistant"` | AI | AI 的文字回复，或者 AI 的**工具调用请求**（此时 content 为 None，有 tool_calls） |
| `"tool"` | 代码 | **工具执行结果**，必须有 `tool_call_id` 跟对应的 tool_call 配对 |

---

## 五、核心代码模式

### 模式 1：Function Calling 标准流程

```python
# ① 定义工具菜单
tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询天气",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名"}},
            "required": ["city"],
        },
    },
}]

# ② 发送消息 + 工具列表
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tools,
)

# ③ 判断 AI 是要调工具还是直接回答
msg = response.choices[0].message
if msg.tool_calls is None:
    return msg.content  # 直接回答
else:
    # ④ 执行工具
    for tc in msg.tool_calls:
        args = json.loads(tc.function.arguments)
        result = TOOL_MAP[tc.function.name](**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
    # ⑤ 继续循环，把结果发给 AI
```

### 模式 2：ReAct 循环

```python
while step < MAX_STEPS:
    step += 1
    response = client.chat.completions.create(model=..., messages=messages, tools=tools)
    msg = response.choices[0].message

    if msg.tool_calls is None:
        return msg.content  # 循环结束

    # 执行工具
    for tc in msg.tool_calls:
        result = TOOL_MAP[tc.function.name](**args)
        messages.append({"role": "tool", ...})

# 超出步数 → 强制总结
```

### 模式 3：Memory 裁剪（按轮丢弃）

```python
def trim_messages(messages, max_tokens):
    # 按"轮"分组（每轮从 user 消息开始）
    rounds = []
    for msg in other_msgs:
        if msg["role"] == "user" and current_round:
            rounds.append(current_round)
            current_round = []
        current_round.append(msg)

    # 从最新轮往前保留
    for r in reversed(rounds):
        if kept_tokens + r_tokens <= max_tokens:
            kept_rounds.insert(0, r)
        else:
            break  # 放不下，更早的轮全丢
```

### 模式 4：Pydantic 对象 → dict 转换

```python
def to_dict(msg):
    """OpenAI 返回的是 Pydantic 对象，转成 dict 方便后续处理"""
    if isinstance(msg, dict):
        return msg
    d = {"role": msg.role}
    if msg.content: d["content"] = msg.content
    if msg.tool_calls: d["tool_calls"] = [...]
    if hasattr(msg, "tool_call_id"): d["tool_call_id"] = msg.tool_call_id
    return d
```

### 模式 5：安全 eval（计算器）

```python
def calculate(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, math.__dict__)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"
```

---

## 六、本周踩过的坑

| 坑 | 原因 | 解决 |
|----|------|------|
| 存储 AI 返回的 msg 对象到 messages | Pydantic 对象不能 `.get("content")` | 用 `to_dict()` 统一转换 |
| Memory 裁剪后 API 报错 400 | tool 消息被保留了但配对的 tool_calls 被丢了 | 按"轮"裁剪，整轮保留或整轮丢弃 |
| 字符串搜索找不到结果 | "Python 创造者 是谁" 不包含子串 "Python" | 改用语义搜索（embedding 相似度） |
| Ollama 503 | Ollama 进程挂了 | 重启 Ollama |

---

## 七、本周脚本总览

| 脚本 | 核心概念 | 新增能力 |
|------|---------|---------|
| `07_function_calling.py` | Tool Calling | AI 能调用工具做事 |
| `08_agent_loop.py` | ReAct + 语义搜索 | 多步推理、自主规划搜索策略 |
| `09_agent_memory.py` | Token 计数 + 按轮裁剪 | 管理对话记忆，防止 token 爆炸 |
