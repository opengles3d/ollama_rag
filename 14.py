from typing import Annotated, TypedDict
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    temperature=0.1,
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

generated_prompt  = SystemMessage(
    """
    You are a essay assistant tasked with writing excellent 3-paragraph essays.""" 
    """
    Generate the best essay possible for the user's request.
    """
    """If the user provides critique, respond with a revised version of your previous attempt. """
)

def generate(state: State) -> State:
    answer = model.invoke([generated_prompt] + state["messages"])
    return {"messages": [answer]}

reflection_prompt = SystemMessage(
    """
    You are a teacher grading an essay submission.Generate critique and recomendations for the user's submission.
    """
    """Provide a detailed recomendayions, including request for length, depth, style, etc."""
)

def reflect(state: State) -> State:
    cls_map = { AIMessage: HumanMessage, HumanMessage: AIMessage }
    translated = [reflection_prompt, state["messages"][0]] + [cls_map[msg.__class__](content=msg.content) for msg in state["messages"][1:]]
    answer = model.invoke(translated)
    return {"messages": [HumanMessage(content = answer.content)]}

def should_continue(state: State):
    if len(state["messages"]) > 6:
        return END
    else:
        return "reflect"

builder = StateGraph(State)
builder.add_node("generate", generate)
builder.add_node("reflect", reflect)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")

graph = builder.compile()

input = {
    "messages": [
        HumanMessage("Write an essay about the importance of education."),
    ]
}

for c in graph.stream(input):
    print(c)


