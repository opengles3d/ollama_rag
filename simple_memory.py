from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all questions to the best of         your ability."),
    ("placeholder", "{messages}"),
])

model = ChatOpenAI(temperature=0, model="qwen3:4b", base_url="http://localhost:11434/v1", api_key="123")

chain = prompt | model

response = chain.invoke({
    "messages": [
        ("human", "Translate this sentence from English to French: I love programming."),
        ("ai", "J'adore programmer."),
        ("human", "What did you just say?"),
    ],
})

print(response.content)
