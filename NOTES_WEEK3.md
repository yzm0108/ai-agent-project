# 学习笔记 — Week 3（MCP 协议 + Agentic RAG）

---

## 一、本周架构全景图

```
10_mcp_weather_server   独立的 MCP 服务进程，提供天气 + 计算 + 时间工具
10_mcp_agent            Agent 通过 MCP Client 连接 Server，自动发现并调用工具
    │
    │  架构变化：工具不再硬编码在 Agent 里
    │  Agent ←→ MCP Client ←→ stdio ←→ MCP Server（独立进程）
    │
    ▼
11_agentic_rag          Agent 自主决定搜索策略
                          要不要搜？搜什么关键词？搜到没有？不够再搜！
```

**核心思想**：
- MCP：工具和 Agent **解耦**，工具提供方独立开发、独立部署
- Agentic RAG：把 RAG 的"搜一次就答"升级为 Agent 的"自主规划搜索策略"

---

## 二、术语中英对照表（Week 3 新增）

| 英文 | 中文 | 一句话解释 |
|------|------|-----------|
| MCP (Model Context Protocol) | 模型上下文协议 | Anthropic 提出的标准化协议，让 AI 连接外部工具和数据源 |
| MCP Server | MCP 服务端 | 提供工具的独立进程，通过 stdio 或 HTTP 与客户端通信 |
| MCP Client | MCP 客户端 | 连接 Server，发现工具、调用工具 |
| stdio (Standard I/O) | 标准输入输出 | MCP 的一种传输方式，通过进程的标准输入/输出通信 |
| JSON-RPC | JSON 远程过程调用 | MCP 底层用的通信协议 |
| Agentic RAG | 智能体增强检索 | 在 RAG 上加入 Agent 决策：自主改写 query、判断是否足够、多轮检索 |
| Query Rewriting | 查询改写 | Agent 把用户问题改写成更适合搜索的关键词 |
| Parallel Tool Call | 并行工具调用 | 一轮同时调多个工具（如同时搜 Python 和 Docker） |
| Tool Discovery | 工具发现 | Agent 启动时自动从 MCP Server 获取可用工具列表 |

---

## 三、新学到的库和函数

### 1. `mcp` — MCP Python SDK

```python
# 服务端
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# 客户端
from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters
```

#### 服务端 API

| 用法 | 作用 |
|------|------|
| `Server("name")` | 创建一个 MCP 服务器 |
| `@server.list_tools()` | 装饰器，注册"列出工具"的处理函数，返回 `list[Tool]` |
| `@server.call_tool()` | 装饰器，注册"调用工具"的处理函数，返回 `list[TextContent]` |
| `Tool(name=..., description=..., inputSchema=...)` | 定义一个工具（名称、描述、参数 schema） |
| `TextContent(type="text", text=...)` | 工具返回的文字结果 |
| `stdio_server()` | 创建 stdio 传输通道（标准输入/输出） |
| `server.run(read_stream, write_stream, options)` | 启动服务器，监听请求 |

#### 客户端 API

| 用法 | 作用 |
|------|------|
| `StdioServerParameters(command="python", args=[...])` | 指定如何启动服务端进程 |
| `stdio_client(params)` | 连接到 stdio 服务端，返回读写流 |
| `ClientSession(read, write)` | 创建 MCP 客户端会话 |
| `session.initialize()` | 初始化握手 |
| `session.list_tools()` | 发现服务端有哪些工具，返回 `mcp_tools.tools` 列表 |
| `session.call_tool(name, arguments)` | 调用工具，返回 `CallToolResult` |
| `result.content` | 工具返回的内容列表（`list[TextContent]`） |
| `c.text` | 从 TextContent 中取出文字 |

### 2. `asyncio` — Python 自带的异步库

```python
import asyncio
```

| 用法 | 作用 |
|------|------|
| `async def func():` | 定义异步函数 |
| `await func()` | 等待异步函数执行完成 |
| `asyncio.run(main())` | 入口：运行异步主函数 |
| `async with ... as x:` | 异步上下文管理器（自动打开/关闭连接） |

### 3. `chromadb` 补充 — 语义搜索集成到 Agent

| 用法 | 作用 |
|------|------|
| `collection.query(query_embeddings=[vec], n_results=k, include=["documents","distances"])` | 向量相似度搜索 |
| `results["distances"][0]` | 距离列表（越小越相似，注意不是余弦相似度） |
| `1.0 - distance` 约等于余弦相似度 | ChromaDB 的 distance 转相似度的近似方法 |

---

## 四、核心概念对比

### MCP vs 硬编码工具

| | 硬编码（07-09） | MCP（10） |
|---|---|---|
| 工具定义在哪 | Agent 代码里 | 独立的 Server 进程 |
| 新增工具 | 改 Agent 代码 | 改 Server 代码，Agent 无需改动 |
| 工具发现 | 写死的 | 运行时自动发现（`list_tools()`） |
| 部署 | Agent 和工具绑定 | Agent 和工具独立部署 |
| 适用场景 | 简单原型 | 生产环境、多服务协作 |

### Agentic RAG vs 普通 RAG

| | 普通 RAG（05, 06） | Agentic RAG（11） |
|---|---|---|
| 检索次数 | 固定 1 次 | 不限次数 |
| Query 改写 | 不做 | Agent 自主改写关键词 |
| 信息不足时 | 硬着头皮答 | 换关键词再搜 |
| 复杂问题 | 整体检索 | 拆开检索（并行多个搜索） |
| 不需要检索时 | 照样检索 | Agent 判断跳过 |

---

## 五、核心代码模式

### 模式 1：MCP Server 标准结构

```python
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

server = Server("my-service")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [Tool(name="...", description="...", inputSchema={...})]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "tool_a":
        result = do_something(arguments)
    return [TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
```

### 模式 2：MCP Client 连接 + 工具格式转换

```python
# ① 启动服务端进程并连接
server_params = StdioServerParameters(command="python", args=["server.py"])
async with stdio_client(server_params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()

        # ② 发现工具
        mcp_tools = await session.list_tools()

        # ③ MCP 工具 → OpenAI Function Calling 格式
        openai_tools = []
        for t in mcp_tools.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema or {"type": "object", "properties": {}},
                },
            })

        # ④ AI 决定调工具 → 通过 MCP 执行
        result = await session.call_tool("tool_name", {"param": "value"})
        # ⑤ 取出文字结果
        text = ""
        for c in result.content:
            if hasattr(c, "text"):
                text += c.text
```

### 模式 3：Agentic RAG——AI 自主搜索策略

核心在 System Prompt 里的指令：
```
1. 先思考需不需要搜（闲聊不搜）
2. 把问题拆成关键词（不要整句话丢进去）
3. 搜到不够 → 换关键词再搜
4. 复杂问题分多次搜索不同方面
5. 多次搜索都找不到 → 诚实告知
6. 够了就整合信息给出完整回答
```

---

## 六、本周脚本总览

| 脚本 | 核心概念 | 新增能力 |
|------|---------|---------|
| `10_mcp_weather_server.py` | MCP Server 工具注册 | 工具独立部署，标准化接口 |
| `10_mcp_agent.py` | MCP Client + 工具发现 | Agent 不再硬编码工具，运行时自动发现 |
| `11_agentic_rag.py` | Agentic RAG | Agent 自主规划搜索策略，多轮检索 |

---

## 七、11 个脚本完成后的技术栈全景

```
你的 GitHub 仓库现在包含：

基础设施层：
  01-03    LLM API 调用、多轮对话、Token 理解

数据层：
  04-06    Embedding、RAG、ChromaDB 持久化

Agent 层：
  07-09    Function Calling、ReAct 循环、Memory 管理

协议层：
  10       MCP 协议（Server + Client）

应用层：
  11       Agentic RAG（智能检索）
```

面试时你可以说：**"我从零搭建了一个完整的 AI Agent 系统，包括 LLM 调用、RAG 检索、Function Calling、ReAct 循环、Memory 管理、MCP 协议集成、Agentic RAG，总共 11 个渐进式模块。"**
