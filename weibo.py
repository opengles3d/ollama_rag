from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="123",
)

def generate_weibo(topic: str):
    response = client.chat.completions.create(
        model="qwen3:4b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Generate a Weibo post about {topic}.Include relevant emojis.一定使用中文回复，不要使用英文。"},
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


content = generate_weibo("Lenovo")  # Example usage
print(content)  # Output the generated Weibo post