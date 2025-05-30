from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchian_ollama import ChatOllamaEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated

model_low_temp = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    temperature=0.1,
)

model_high_temp = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    temperature=0.7,
)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    user_query: str
    sql_query: str
    sql_explaination: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    sql_query: str
    sql_explaination: str

generate_prompt = SystemMessage(
    """
    You are helpful data analyst who gernerates SQL queries for users based on their natural language questions.
    """
)

def generate_sql(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [generate_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    return {
        "sql_query": res.content,
        "messages": [user_message, res],
    }

explain_prompt = SystemMessage(
    """
    You are a SQL expert who explains SQL queries to users based on their natural language questions.
    """
)

def explain_sql(state: State) -> State:
    messages = [explain_prompt, *state["messages"]]
    res = model_high_temp.invoke(messages)
    return {
        "sql_explaination": res.content,
        "messages": res,
    }

builder = StateGraph(State, input=Input, output=Output)
builder.add_node("generate_sql", generate_sql)
builder.add_node("explain_sql", explain_sql)
builder.add_edge(START, "generate_sql")
builder.add_edge("generate_sql", "explain_sql")
builder.add_edge("explain_sql", END)

graph = builder.compile()

r = graph.invoke({
    "user_query": "What is the total sales for each product?"})

print(r["sql_query"])

