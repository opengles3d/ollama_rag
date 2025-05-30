from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API 地址
    api_key="ollama"  # Ollama 默认无需真实 API Key，填任意值即可
)

def get_embedding(text, model="nomic-embed-text:latest"):
    if not text.strip():
        return None
    response = client.embeddings.create(
        input=text[:8192*2],  # Truncate to approximate token limit
        model=model,
        dimensions=1024
    )
    return response.data[0].embedding

text_embedding = get_embedding("你好，什么是大模型？")
print(text_embedding)


# 发送请求
response = client.chat.completions.create(
    model="sammcj/qwen2.5-coder-7b-instruct:q8_0",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "你好，什么是大模型？"}
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512    # 最大生成 token 数
)

# 打印结果
print(response.choices[0].message.content)

response = client.completions.create(
    model="sammcj/qwen2.5-coder-7b-instruct:q8_0",  # 指定模型
    #prompt="def fib(a):",
    #suffix="    return fib(a-1) + fib(a-2)",
    prompt="朝辞白帝彩云间，",
    suffix="两岸猿声啼不住，轻舟已过万重山。",
    max_tokens=128
)
print(response.choices[0].text)