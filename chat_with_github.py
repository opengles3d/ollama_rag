#pip install PyGithub
from github import Github
import logging
import os
#pip install streamlit
import streamlit as st
from openai import OpenAI
import json

#streamlit run .\chat_with_github.py
#run this script with streamlit, it will start a web server and you can access it via http://localhost:8501

def get_repo_tree(repo_full_name, branch=None):
    # 从环境变量里读取github的token
    #token = os.getenv("GITHUB_TOKEN", None)
    # 创建一个github对象
    #g = Github(token)
    g = Github()  # 使用无认证的方式获取公共仓库信息
    # 获取一个repo对象
    repo = g.get_repo(f"{repo_full_name}")
    # 获取repo的目录结构
    # Ensure branch is not None
    if branch is None:
        branch = "master"
    tree = repo.get_git_tree(sha=branch, recursive=True)
    # 构造树形式的字符串
    tree_str = ""
    for item in tree.tree:
        tree_str += f"{item.path}\n"
    return tree_str

#tree = get_repo_tree("ollama/ollama", "main")
#print(tree)

def get_repo_file_content(repo_full_name, file_path, branch=None):
    # 从环境变量里读取github的token
    #token = os.getenv("GITHUB_TOKEN", None)
    # 创建一个github对象
    #g = Github(token)
    g = Github()  # 使用无认证的方式获取公共仓库信息
    # 获取一个repo对象
    repo = g.get_repo(f"{repo_full_name}")
    # 获取文件内容
    # Ensure branch is not None
    if branch is None:
        branch = "master"
    file_content = repo.get_contents(file_path, ref=branch)
    return file_content.decoded_content.decode("utf-8")

#file_content = get_repo_file_content("ollama/ollama", "README.md", "main")
#print(file_content)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_repo_tree",
            "description": "获取指定GitHub仓库的目录结构",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_full_name": {
                        "type": "string",
                        "description": "GitHub仓库的全名，例如 'owner/repo'",
                    },
                    "branch": {
                        "type": "string",
                        "description": "指定分支名称，默认为 'master'",
                    },
                },
                "required": ["repo_full_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repo_file_content",
            "description": "获取指定GitHub仓库中某个文件的内容",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_full_name": {
                        "type": "string",
                        "description": "GitHub仓库的全名，例如 'owner/repo'",
                    },
                    "file_path": {
                        "type": "string",
                        "description": "文件在仓库中的路径，例如 'path/to/file.txt'",
                    },
                    "branch": {
                        "type": "string",
                        "description": "指定分支名称，默认为 'master'",
                    },
                },
                "required": ["repo_full_name", "file_path"],
            },
        },
    },
]

available_functions = {
    "get_repo_tree": get_repo_tree,
    "get_repo_file_content": get_repo_file_content,
}

client = OpenAI(
    api_key="123",
    base_url="http://localhost:11434/v1",  # local Ollama server
)

MODEL_NAME = "qwen3:4b"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Starting Streamlit script ...")

st.set_page_config(
    page_title="Chat with GitHub",
    page_icon="🤖",
    layout="wide",
)

st.subheader("Chat with GitHub - just a demo")

if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "system", "content": "你是一个帮助用户解读Github上代码库的软件专家。"}]
else:
    for message in st.session_state.messages:
        if message["role"] == "user" or message["role"] == "assistant":
            if "content" in message.keys():
                with st.chat_message(message["role"]):
                    st.write(message["content"])

if user_prompt := st.chat_input('在此输入你的问题'):
    logging.info(f"User prompt: {user_prompt}")
    with st.chat_message("user"):
        st.write(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})
    
    with st.chat_message("assistant"):
        with st.spinner("人工智能在思考...."):
            gpt_response = client.chat.completions.create(
                model = MODEL_NAME,
                messages = st.session_state.messages,
                tools = tools
            )

            response_msg = gpt_response.choices[0].message
            tool_calls = response_msg.tool_calls

            if tool_calls:
                st.session_state.messages.append({"role": "assistant", "tool_calls": response_msg.tool_calls})
                for tool_call in tool_calls:
                    function_args = json.loads(tool_call.function.arguments)
                    logging.info(f"Function call: {tool_call.function.name} with arguments: {function_args}")
                    if tool_call.function.name == "get_repo_tree":
                        function_response = available_functions["get_repo_tree"](function_args.get("repo_full_name"), function_args.get("branch"))
                    if tool_call.function.name == "get_repo_file_content":
                        function_response = available_functions["get_repo_file_content"](function_args.get("repo_full_name"), function_args.get("file_path"), function_args.get("branch"))
                    logging.info(f"Function {tool_call.function.name} responsed")
                    st.session_state.messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_call.function.name,
                            "content": function_response,
                        })
                secondary_gpt_response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=st.session_state.messages,
                    )
                secondary_gpt_response_msg = secondary_gpt_response.choices[0].message
                if secondary_gpt_response_msg.role == "assistant":
                    st.write(secondary_gpt_response_msg.content)
                    st.session_state.messages.append({"role": "assistant", "content": secondary_gpt_response_msg.content})
            else:
                if response_msg.role == "assistant":
                    st.write(response_msg.content)
                    st.session_state.messages.append({"role": "assistant", "content": response_msg.content})

st.write("**注意：人工智能生成内容仅供参考。**")
logging.info("Streamlit script ended.")