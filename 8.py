from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
import re

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an assistant. Answer all questions directly and concisely. Do not show your thought process or reasoning steps. Just give the answer.
    """),
    ("placeholder", "{messages}"),
])

model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:14b",
    temperature=0.2,
)

chain = prompt | model
result = chain.invoke({
    "messages" : [
    ("human", "Translate this sentence from English to Chinese: 'Hello, how are you?'"),
    ("ai", "你好吗？"),
    ("human", "What did you just say?")
    ],
})

# 移除<think>...</think>标记及内容，并去除前导换行符
clean_content = re.sub(r"<think>.*?</think>", "", result.content, flags=re.DOTALL)
clean_content = clean_content.lstrip("\n")
print(clean_content)