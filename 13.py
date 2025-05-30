import ast
from typing import Annotated, TypedDict, Literal

from uuid import uuid4


from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolCall
from langchain_ollama import ChatOllama, OllamaEmbeddings
#from langchain_ollama.tools import OllamaEmbeddings


from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_core.documents import Document


@tool
def calculator(query: str) -> str:
    """
    A simple calculator that can do basic arithmetic operations.
    """
    print(f"--------{query}--------")
    try:
        # 只保留数字和运算符，防止非法输入
        allowed = "0123456789+-*/(). "
        safe_query = "".join(c for c in query if c in allowed)
        result = eval(safe_query, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def baidu_search_tool(query: str) -> str:
    """
    Search Baidu and return the first result URL.
    """
    return """Born in Plymouth, Vermont, on July 4, 1872, Coolidge was the son of a village storekeeper. He was graduated from Amherst College with honors, and entered law and politics in Northampton, Massachusetts. Slowly, methodically, he went up the political ladder from councilman in Northampton to Governor of Massachusetts, as a Republican. En route he became thoroughly conservative.Coolidge died suddenly of coronary thrombosis at The Beeches on January 5, 1933, at 12:45 p.m.""".strip()
# 替换 tools
tools = [baidu_search_tool, calculator]

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    temperature=0
    ).bind_tools(tools)

embeddings = OllamaEmbeddings(
    base_url="http://localhost:11434",
    model="qwen3:4b"
    )

tools_retriever = InMemoryVectorStore.from_documents(
    [Document(tool.description, metadata={"name": tool.name}) for tool in tools],
    embeddings
).as_retriever()


class State(TypedDict):
    messages : Annotated[list, add_messages]
    selected_tools: list[str]

def model_node(state: State) -> State:
    selected_tools = [tool for tool in tools if tool.name in state["selected_tools"]]
    res = model.bind_tools(selected_tools).invoke(state["messages"])
    return {"messages": res}

def select_tools(state: State) -> State:
    query = state["messages"][-1].content
    tool_docs = tools_retriever.invoke(query)
    return {
        "selected_tools": [doc.metadata["name"] for doc in tool_docs],}

def first_model(state: State) -> State:
    query = state["messages"][-1].content
    search_tool_call = ToolCall(
        name="baidu_search_tool",
        args={"query": query},
        id = uuid4().hex,
    )
    return {
        "messages": AIMessage(content="", tool_calls=[search_tool_call]),    
    }


builder = StateGraph(State)
builder.add_node("select_tools", select_tools)
builder.add_node("model", model_node)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "select_tools")
builder.add_edge("select_tools", "model")
builder.add_conditional_edges("model", tools_condition)
builder.add_edge("tools", "model")

graph = builder.compile()

input = {
    "messages": [
        HumanMessage("""How old was the 30th president of the United States when he died? Please use the calculator tool to calculate the answer based on the information from search tool."""),
    ]
}

for c in graph.stream(input):
    print(c)

