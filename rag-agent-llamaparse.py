from dotenv import load_dotenv
import time

load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_parse import LlamaParse

def multiply(a: float, b: float) -> float:
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)

def add(a: float, b: float) -> float:
    return a + b

add_tool = FunctionTool.from_defaults(fn=add)

# Start timing
start_time = time.time()

Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.0)

dccuments2 = LlamaParse(result_type="markdown").load_data(
    "./data/2023_canadian_budget.pdf")
index2 = VectorStoreIndex.from_documents(dccuments2)
query_engine2 = index2.as_query_engine()


budget_tool = QueryEngineTool.from_defaults(
    query_engine2,
    name="canadian_budget_2023",
    description="A RAG engine with some basic facts about the 2023 canadian federal budget"
)

agent = ReActAgent.from_tools(
    tools=[multiply_tool, add_tool, budget_tool],
    verbose=True,
)

# End timing for setup
setup_end_time = time.time()

# response2 = agent.chat('What is the total amount of the 2023 Canadian federal budget multiplied by 3?')
response2 = query_engine2.query('How much exactly was allocated to a tax credit to promote investment in green technologies in the 2023 Canadian federal budget?')

# End timing for query
query_end_time = time.time()

print(response2)

# Print elapsed times
print(f"Setup time: {setup_end_time - start_time} seconds")
print(f"Query time: {query_end_time - setup_end_time} seconds")
print(f"Total time: {query_end_time - start_time} seconds")