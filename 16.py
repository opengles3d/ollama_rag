from typing import Literal
from langchain_ollama import ChatOllama
from pydantic import BaseModel
from langgraph.graph import StateGraph, END, START

class SupervisorDecision(BaseModel):
    next: Literal ["researcher", "coder", "FINISH"]

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b"
)

model = model.with_structured_output(SupervisorDecision)

agents = ["researcher", "coder"]

system_prompt_part_1 = f"""You are a supervisor tasked with managing a converstion between the flollowing workers:{agents}. Given the fllowing user request, respond with the worker to act next.Each worker will perform a task and respond with their results and status. when finished, respond with FINISH."""

system_prompt_part_2 = f"""Given the converstion above, who should act next? Or should we FINISH? Select one of : {','.join(agents)}, FINISH. """

def supervisor(state):
    messages = [
        ("system", system_prompt_part_1),
        *state["messages"],
        ("system", system_prompt_part_2)
    ]

    return model.invoke(messages) 

class AgentState(BaseModel):
    next: Literal["researcher", "coder", "FINISH"]

def researcher(state: AgentState):
    response = model.invoke(...)
    return {"messages": [response] }

def coder(state: AgentState):
    response = model.invoke(...)
    return {"messages": [response] }

builder = StateGraph(AgentState)
builder.add_node(supervisor)
builder.add_node(researcher)
builder.add_node(coder)

builder.add_edge(START, "supervisor")
builder.add_conditional_edges("supervisor", lambda state: state["next"])
builder.add_edge("researcher", "supervisor")
builder.add_edge("coder", "supervisor")

supervisor = builder.compile()


