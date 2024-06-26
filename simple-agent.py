from dotenv import load_dotenv

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool

def multiply(a: float, b: float) -> float:
    return a * b


multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool],
    llm=llm,
    verbose=True,
)

response = agent.chat('What is 20+(2*4)? Use a tool to calculate every step.')

print(response)