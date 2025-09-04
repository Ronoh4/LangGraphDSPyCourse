# Basic Principles 
# 1. LangGraph is Typed meaning it expects a State schema that must be defined using python typing
# 2. Each node and edge in the graph expects and returns values conforming to the schema. 
# 3. Because it is stateful, LangGraph uses Annotated to add rules and metadata functions like reducers to state schemas. 
# 4. StateGraph, or Graph defines the structure of Nodes and Edges. 

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Define state
class State(TypedDict):
    messages: Annotated[list, add_messages]

graph = StateGraph(State)

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_base_url = os.getenv("GROQ_BASE_URL")

# Initialize Groq LLM
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    base_url=groq_base_url,
    temperature=0.5,
)

# -------- Nodes -------- #

# Node 1: first LLM response to user input
def FirstResponseNode(state: State):
    response = llm.invoke(state["messages"])
    print("\n[Raw Node1 Response]:\n", response.content, "\n")  # show raw reply
    return {"messages": [response]}

# Node 2: user chooses how to transform Node1’s response
def TransformNode(state: State):
    last_ai_msg = state["messages"][-1]  # last AI message (Node1 output)

    # Ask user in terminal what to do with Node1's reply
    transform_instr = input("Transform Node1 response: ")

    # Feed instruction + Node1 output together
    user_msg = {
        "role": "user",
        "content": f"{transform_instr}\n\n{last_ai_msg.content}"
    }
    transformed = llm.invoke([user_msg])
    return {"messages": [transformed]}

# -------- Graph -------- #
graph.add_node("node1_response", FirstResponseNode)
graph.add_node("node2_transform", TransformNode)

graph.add_edge(START, "node1_response")
graph.add_edge("node1_response", "node2_transform")
graph.add_edge("node2_transform", END)

build = graph.compile()

# -------- Loop -------- #
while True:
    user_input = input("\nUser: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Run through graph
    result = build.invoke({"messages": [{"role": "user", "content": user_input}]})
    final_msg = result["messages"][-1]

    if final_msg.type == "ai":
        print("\n[Transformed Response]:\n", final_msg.content, "\n")
