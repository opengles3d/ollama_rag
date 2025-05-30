from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, List

# 初始化 LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:14b",
    temperature=0.2,
)

# 节点函数，输入和输出都是 state 字典
def llm_node(state):
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": messages, "response": response}

# 构建 LangGraph，使用 state_schema 参数
class MyState(TypedDict):
    messages: List[HumanMessage]
    response: object

graph = StateGraph(state_schema=MyState)
graph.add_node("llm", llm_node)
graph.add_edge("llm", END)
graph.set_entry_point("llm")
graph.set_finish_point(END)
graph = graph.compile()

# 运行一次
inputs = {"messages": [HumanMessage(content="你好，请介绍一下你自己")]}
result = graph.invoke(inputs)
print(result["response"])
