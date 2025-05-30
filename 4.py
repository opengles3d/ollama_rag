from openai import OpenAI
client = OpenAI(
    api_key="ec2f6f87-80bd-402b-9b9b-a066cb3e27b7", # ModelScope Token
    base_url="https://api-inference.modelscope.cn/v1"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-VL-72B-Instruct", # ModleScope Model-Id
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://modelscope.oss-cn-beijing.aliyuncs.com/demo/images/bird-vl.jpg"}
                },
                {   "type": "text", 
                    "text": "Count the number of birds in the figure, including those that are only showing their heads. To ensure accuracy, first detect their key points, then give the total number."
                },
            ],
        }
    ],
    stream=False
    )

print(response)