from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document
from langchain_core.vectorstores.in_memory import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings

from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Literal

embedings = OllamaEmbeddings(
    base_url="http://localhost:11434", 
    model="qwen3:4b",
    temperature=0.1)

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
    domain: Literal["records", "insurance"]
    documents: list[Document]
    answer: str

class Input(TypedDict):
    user_query: str

class Output(TypedDict):
    documents: list[Document]
    answer: str

medical_records_store = InMemoryVectorStore.from_documents([], embedings)
medical_records_retriever =  medical_records_store.as_retriever()

insurance_store = InMemoryVectorStore.from_documents([], embedings)
insurance_retriever = insurance_store.as_retriever()

route_prompt = SystemMessage(
    """
    You need to decide which domain to route the user query to. You have two domains to choose from:
    - records: contains medical records of the patients.
    - insurance: contains insurance information of the patients.
    """
)

def route_node(state: State) -> State:
    user_message = HumanMessage(state["user_query"])
    messages = [route_prompt, *state["messages"], user_message]
    res = model_low_temp.invoke(messages)
    state["domain"] = res.content
    return {
        "messages": [user_message, res],
        "domain": res.content,
    }

def pick_retriever(state: State) -> Literal["retrieve_medical_records", "retrieve_insurance_faqs"]:
    if state["domain"] == "records":
        return "retrieve_medical_records"
    elif state["domain"] == "insurance":
        return "retrieve_insurance_faqs"
    else:
        raise ValueError("Invalid domain")

def retrieve_medical_records(state: State) -> State:
    documents = medical_records_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }

def retrieve_insurance_faqs(state: State) -> State:
    documents = insurance_retriever.invoke(state["user_query"])
    return {
        "documents": documents,
    }

medical_records_prompt = SystemMessage(
    """
    You are a helpful medical chatbot who answers questions based on the patient's medical records, such as diagnosis, treament, and prescription.
    """
)

insurance_faqs_prompt = SystemMessage(
    """
    You are a helpful insurance chatbot who answers questions based on the patient's insurance information, such as coverage, claims, and benefits.
    """
)

def generate_answer(state: State) -> State:
    if state["domain"] == "records":
        messages = [medical_records_prompt, *state["messages"], HumanMessage(f"Documents: {state["documents"]}")]
    elif state["domain"] == "insurance":
        messages = [insurance_faqs_prompt, *state["messages"], HumanMessage(f"Documents: {state["documents"]}")]
    else:
        raise ValueError("Invalid domain")
    
    res = model_high_temp.invoke(messages)
    return {
        "answer": res.content,
        "messages": res,
    }

builder = StateGraph(State, input=Input, output=Output)
builder.add_node("router", route_node)
builder.add_node("retrieve_medical_records", retrieve_medical_records)
builder.add_node("retrieve_insurance_faqs", retrieve_insurance_faqs)
builder.add_node("generate_answer", generate_answer)
builder.add_edge(START, "router")
builder.add_conditional_edges("router", pick_retriever)
builder.add_edge("retrieve_medical_records", "generate_answer")
builder.add_edge("retrieve_insurance_faqs", "generate_answer")
builder.add_edge("generate_answer", END)


graph = builder.compile()



