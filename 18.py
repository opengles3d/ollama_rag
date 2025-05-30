from langgraph.graph import StateGraph, START, END
from langchain_community.chat_models import ChatOllama

# 定义 input 节点，直接返回输入
def input_node(state):
    return state

# 定义 ChatOllama 节点
def chat_node(state):
    model = ChatOllama(
        base_url="http://localhost:11434",
        model="qwen3:4b"
    )
    # 假设 state 里有 "user_input"
    response = model.invoke([{"role": "user", "content": state["user_input"]}])
    return state

# 构建 langgraph
builder = StateGraph(dict)
builder.add_node("input", input_node)
builder.add_node("chat", chat_node)
builder.add_edge(START, "input")
builder.add_edge("input", "chat")
builder.add_edge("chat", END)

graph = builder.compile()

# 运行
#for c in graph.stream({"user_input": "你好，langgraph!"}, stream_mode='debug'):
#    print("当前状态:", c)
import asyncio

output = graph.astream_events({"user_input": "你好，langgraph!"}, version="v2")

async def main():
    async for event in output:
        #print("事件:", event)
        if event["event"] == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                #print("模型输出:", content)
                print(content, end='')

asyncio.run(main())