import os
import argparse
from openai import OpenAI

parser = argparse.ArgumentParser(description='AI对话脚本')
parser.add_argument('--model', type=str, default='qwen3:14b',
                    help='指定使用的模型名称（默认：qwen3:14b）')
parser.add_argument('--api_key', type=str, default='123',
                    help='指定API密钥（默认使用环境变量API_KEY）')
parser.add_argument('--base_url', type=str, default="http://localhost:11434/v1",
                    help='指定API基础URL（默认：http://localhost:11434/v1）')

args = parser.parse_args()

api_key = args.api_key if args.api_key else os.getenv('API_KEY')
print(api_key)
print(args.base_url)
client = OpenAI(
    api_key=api_key,
    base_url=args.base_url, 
)


messages = []

while True:
    user_input = input("\n请输入（直接回车退出对话）: ").strip()
    
    if not user_input or user_input.lower() == 'exit':
        break
    
    messages.append({'role': 'user', 'content': user_input})
    
    reasoning_content = ""
    answer_content = ""
    is_answering = False
    has_reasoning = False
    
    completion = client.chat.completions.create(
        model=args.model,
        messages=messages,
        stream=True,
        temperature=0.7
    )
    
    for chunk in completion:
        if not chunk.choices:
            if hasattr(chunk, 'usage'):
                pass
        else:
            delta = chunk.choices[0].delta
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content is not None:
                if not has_reasoning:
                    has_reasoning = True
                    print(f"\n{args.model}正在思考...")

                print(f"\033[37m{delta.reasoning_content}\033[0m", end='', flush=True)
                reasoning_content += delta.reasoning_content
               
            elif hasattr(delta, 'content') and delta.content is not None:
                if not is_answering and delta.content != '':
                    if has_reasoning:
                        print(f"\n\n{args.model} 回复:")
                    else:
                        print(f"\n{args.model} 回复:")
                    is_answering = True

                print(f"{delta.content}", end='', flush=True)
                answer_content += delta.content
              
    if answer_content:
        messages.append({'role': 'assistant', 'content': answer_content})
        print()

print("对话结束")        
