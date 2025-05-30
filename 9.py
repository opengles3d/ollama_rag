from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages, AIMessage, filter_messages, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
import uuid
import datetime

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b",
    temperature=0.2,
)


def dummy_token_counter(messages: list[BaseMessage]) -> int:
    # treat each message like it adds 3 default tokens at the beginning
    # of the message and at the end of the message. 3 + 4 + 3 = 10 tokens
    # per message.

    default_content_len = 4
    default_msg_prefix_len = 3
    default_msg_suffix_len = 3

    count = 0
    for msg in messages:
        if isinstance(msg.content, str):
            count += default_msg_prefix_len + default_content_len + default_msg_suffix_len
        if isinstance(msg.content, list):
            count += default_msg_prefix_len + len(msg.content) *  default_content_len + default_msg_suffix_len
    return count

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=dummy_token_counter,
    include_system=True,
    allow_partial=False,
    start_on="hunman"
)

messages = [
    SystemMessage(content="you are a helpful assistant."),
    HumanMessage(content="hi, I'm Bob"),
    AIMessage(content="hi"),
    HumanMessage(content="I like money."),
    AIMessage(content="nice."),
    HumanMessage(content="What's 2 + 2?"),
    AIMessage(content="4"),
    HumanMessage(content="Thanks."),
    AIMessage(content="You're welcome."),
    HumanMessage(content="Have fun?"),
    AIMessage(content="yes"),
]

r = trimmer.invoke(messages)
print(r)

filter = filter_messages(messages, include_types="human")
print(filter)

class State(TypedDict):
    messages: Annotated[list, add_messages]

def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}

builder = StateGraph(state_schema=State)
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

#graph = builder.compile()
graph = builder.compile(checkpointer=MemorySaver())
#graph.get_graph().draw_mermaid("graph.png")
#print(graph.get_graph().draw_mermaid())

#from IPython.display import Image, display
#from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

#display(Image(graph.get_graph().draw_mermaid_png()))
config = {
        "configurable": {
            "thread_id": uuid.uuid4(),
            "thread_ts": datetime.datetime.now(datetime.UTC)
        }
    }
inputs = {"messages": [HumanMessage("你好，请介绍一下你自己")]}
for chunk in graph.stream(inputs, config=config):
    print(chunk["chatbot"]["messages"][0].content)
#result = graph.invoke(inputs)
#print(result["messages"][0].content)

thread1 = {"configurable": {
            "thread_id": "1"
        }
    }

inputs1 = {"messages": [HumanMessage("hi, my name is Jack")]}
for chunk in graph.stream(inputs1, config=thread1):
    print(chunk["chatbot"]["messages"][0].content)

thread2 = {"configurable": {
            "thread_id": "1"
        }
    }

inputs2 = {"messages": [HumanMessage("What is my name?")]}
for chunk in graph.stream(inputs2, config=thread2):
    print(chunk["chatbot"]["messages"][0].content)

state1 = graph.get_state(thread1)
print(state1)

graph.update_state(state1.config, values={"messages" : [HumanMessage("I love LLM")]})