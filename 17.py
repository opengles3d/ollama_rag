from pydantic import BaseModel, Field

class Joke(BaseModel):
    setup: str = Field(description="The setup of the joke")
    punchline: str = Field(description="The punchline of the joke")

from langchain_ollama import ChatOllama
model = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:4b"
)

model = model.with_structured_output(Joke)

res = model.invoke("Tell me a jokea about cat")
print(res)