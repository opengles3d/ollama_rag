from langchain_ollama import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# 自定义计算函数
def my_calculator(query: str) -> str:
    try:
        # 仅做简单的 eval，实际生产环境请用更安全的解析
        result = eval(query, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

# 定义自定义 Tool
my_calc_tool = Tool(
    name="Calculator",
    func=my_calculator,
    description="用于进行数学计算。输入如 '123 * 456 + 789'。"
)

# 初始化 Ollama LLM
llm = ChatOllama(
    base_url="http://localhost:11434",
    model="qwen3:14b",
    temperature=0.2,
)

tools = [my_calc_tool]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
)

result = agent.invoke("计算 123 * 456 并加上 789")
print(result)
