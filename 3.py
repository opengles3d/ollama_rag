from langchain_community.chat_models import ChatOllama
import base64

# 读取本地图片并进行base64编码
with open("d:\\temp\\2.png", "rb") as img_file:
    base64_image = base64.b64encode(img_file.read()).decode("utf-8")

llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen2.5vl:7b",
    temperature=0.2,
)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                #"text": "详细描述图片的内容"
                "text": "Extract the table from the image and return it in Markdown format. If no table is preset, retuen an empty string. Do not include additional explainations."
            },
            {
                "type": "image_url",
                "image_url": f"data:image/png;base64,{base64_image}"
            }
        ]
    }
]

response = llm.invoke(messages)
print(response)
table_markdown = response.content