import os
from dotenv import load_dotenv
from openai import OpenAI
import chromadb

load_dotenv()

ollama = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
deepseek = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL"),
)

# ============ 连接 ChromaDB ============
chroma = chromadb.PersistentClient(path="./chroma_db")
collection = chroma.get_or_create_collection(name="my_knowledge")


# ============ 工具函数：只写一次，到处复用 ============
def get_embedding(text):
    """任何文字 → 向量。所有地方都用这个函数调，不重复写。"""
    resp = ollama.embeddings.create(model="bge-embed", input=text)
    return resp.data[0].embedding

# ============ 首次运行初始化知识库 ============
if collection.count() == 0:
    print("知识库为空，正在初始化...")
    documents = [
        "贵阳是贵州省的省会，位于中国西南部，气候凉爽，被称为'中国避暑之都'。",
        "贵州茅台是中国最著名的白酒品牌之一，原产地在贵州省遵义市的茅台镇。",
        "Python 是由 Guido van Rossum 于 1991 年创造的编程语言，以简洁易读著称。",
        "大语言模型（LLM）是通过海量文本训练的神经网络，代表包括 GPT、Claude 和 DeepSeek。",
        "黄果树瀑布位于贵州省安顺市，是亚洲最大的瀑布之一，高约 77.8 米。",
        "Transformer 架构于 2017 年由 Google 提出，是现代大语言模型的基础结构。",
    ]
    for i, doc in enumerate(documents):
        collection.add(
            ids=[str(i)],
            documents=[doc],
            embeddings=[get_embedding(doc)],
        )
    print(f"共 {collection.count()} 条知识入库\n")
else:
    print(f"知识库已有 {collection.count()} 条记录\n")


def retrieve(question, top_k=2):
    results = collection.query(
        query_embeddings=[get_embedding(question)],
        n_results=top_k,
        include=["documents", "distances"],
    )
    docs = []
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        docs.append((round(1.0 - dist, 4), doc))
    return docs


def answer(question, retrieved_docs):
    context = "\n".join([f"- {doc}" for _, doc in retrieved_docs])
    resp = deepseek.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "基于以下知识库段落回答问题。"
                    f"相关段落：\n{context}\n\n"
                    "严格基于段落回答，没有就说'知识库中没有'。不要编造。"
                ),
            },
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content


# ============ 交互循环 ============
print("=" * 50)
print("命令：直接输入问题 / add 内容 / quit 退出")
print("=" * 50)

while True:
    try:
        user_input = input("\n你: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n再见！")
        break

    if not user_input:
        continue
    if user_input.lower() == "quit":
        print("再见！")
        break

    # 追加知识
    if user_input.startswith("add "):
        new_knowledge = user_input[4:]
        new_id = str(collection.count())
        collection.add(
            ids=[new_id],
            documents=[new_knowledge],
            embeddings=[get_embedding(new_knowledge)],
        )
        print(f"已入库（ID={new_id}）")
        continue

    # 问答
    docs = retrieve(user_input)
    print("检索结果：")
    for sim, doc in docs:
        print(f"  (相似度 {sim:.4f}) {doc[:60]}...")

    ans = answer(user_input, docs)
    print(f"\nAI: {ans}")
