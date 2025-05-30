from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated

class State(TypedDict):
    foo: str

class SubgraphState(TypedDict):
    foo: str
    bar: str

def subgraph_node(state: SubgraphState) :
    return {"foo": state["foo"] + "bar"}

def node(state: State):
    response = subgraph.invoke({"bar": state["foo"]})
    return {"foo": response["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node("subgraph_node", subgraph_node) 

subgraph = subgraph_builder.compile()

builder = StateGraph(State)
builder.add_node("subgraph", subgraph)
builder.add_node(node)

graph = builder.compile()
