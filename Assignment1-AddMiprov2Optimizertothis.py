# This flow uses a basic predict dspy module to generate the query opts
# Basic modules are suboptimal. Here, use an advanced MiproV2 optimizer as part of learning dspy

from typing import Literal
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from firecrawl import Firecrawl  # Firecrawl SDK

# 🔥 DSPy
import dspy


# ----------------- ENV -----------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
groq_base_url = os.getenv("GROQ_BASE_URL")
serp_api_key = os.getenv("SERP_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env file")
if not groq_base_url:
    raise ValueError("GROQ_BASE_URL not found in .env file")
if not serp_api_key:
    raise ValueError("SERP_API_KEY not found in .env file")
if not firecrawl_api_key:
    raise ValueError("FIRECRAWL_API_KEY not found in .env file")


# ----------------- LLM -----------------
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    base_url=groq_base_url,
    temperature=0.5,
)

# ----------------- DSPy CONFIGURATION -----------------
# Configure DSPy with Groq (add this after your existing imports and env setup)
groq_dspy_lm = dspy.LM(
    model="groq/meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    temperature=0.3,  # Lower temperature for consistent optimization
    max_tokens=150,   # Short responses for query optimization
    cache=True
)
dspy.configure(lm=groq_dspy_lm)

# ----------------- Firecrawl -----------------
firecrawl = Firecrawl(api_key=firecrawl_api_key)


# ----------------- TOOLS -----------------
@tool
def knowledge_graph_search(query: str) -> dict:
    """Extract specific information from Google's Knowledge Graph for reliable facts."""
    params = {"engine": "google", "q": query, "api_key": serp_api_key}
    search = GoogleSearch(params)
    results = search.get_dict()
    knowledge_graph = results.get("knowledge_graph", None)
    if knowledge_graph:
        return {"knowledge_graph": knowledge_graph}
    else:
        return {"error": "No Knowledge Graph data found."}


@tool
def organic_google_search(query: str) -> dict:
    """Perform a general organic Google search. Returns top 6 results if found."""
    params = {"engine": "google", "q": query, "api_key": serp_api_key}
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", None)

    if organic_results:
        return {"results": organic_results[:6]}
    else:
        return {"error": "No general search results found."}


@tool
def firecrawl_scrape(url: str) -> dict:
    """Scrape webpage content into a concise summary using Firecrawl."""
    try:
        doc = firecrawl.scrape(url, formats=["summary"])
        return {
            "url": url,
            "summary": getattr(doc, "summary", None),
            "metadata": getattr(doc, "metadata", None),
        }
    except Exception as e:
        return {"error": str(e)}


# Register tools
tools = [knowledge_graph_search, organic_google_search, firecrawl_scrape]
tools_by_name = {t.name: t for t in tools}


# ----------------- LLM WITH TOOLS -----------------
llm_with_tools = llm.bind_tools(tools)


# ----------------- DSPy SIGNATURE -----------------
class ToolQuerySig(dspy.Signature):
    """
    Optimize tool arguments at runtime.
    Input: raw user question or raw tool args.
    Output: improved tool arguments (e.g., better search query or cleaner URL).
    """
    raw_input: str = dspy.InputField(desc="Raw query or argument passed to a tool.")
    optimized_input: str = dspy.OutputField(desc="Enhanced search query that improves searchability with better terminology, relevant keywords, and clearer and more specific phrasing while preserving all original intent and requirements.")


# Initialize DSPy program (zero-shot for now, but can plug in optimizers like MIPRO later)
class ToolQueryOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(ToolQuerySig)

    def forward(self, raw_input: str) -> str:
        return self.predictor(raw_input=raw_input).optimized_input


query_optimizer = ToolQueryOptimizer()


# ----------------- GRAPH NODES -----------------
def llm_call(state: MessagesState):
    """LLM decides whether to call a tool or stop."""
    messages = [SystemMessage(content="""You are a research assistant.

Workflow:
1. Always attempt `knowledge_graph_search` first.  
2. If knowledge graph content is available, use it directly.  
3. If not available or an error occurs, perform `organic_google_search`.  
4. From the search results, carefully select the **most relevant** URL (not just the first one).  
5. Call `firecrawl_scrape` on the chosen URL to extract content.  

Use this workflow strictly in order. Do not skip steps unless explicitly instructed.
""")] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: dict):
    """Executes tool calls made by the LLM, with DSPy optimization of arguments."""
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]

        # 🔒 Safely extract args
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}

        # 🔥 Only optimize if "query" is present - your approach is perfect!
        if "query" in args:
            original_query = args["query"]
            optimized_query = query_optimizer(args["query"])
            args["query"] = optimized_query
            
            # Optional: Add some debug output to see the optimization in action
            print(f"🔥 DSPy Query Optimization:")
            print(f"   Original: '{original_query}'")
            print(f"   Optimized: '{optimized_query}'")

        observation = tool.invoke(args)
        results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
    return {"messages": results}



def should_continue(state: MessagesState) -> Literal["Action", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Action"
    return END


# ----------------- GRAPH -----------------
agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("environment", tool_node)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")

# Attach memory
memory = InMemorySaver()
agent = agent_builder.compile(checkpointer=memory)


# Choose a thread id for this interactive run
THREAD_ID = "chat-session"


# ----------------- INTERACTIVE LOOP -----------------
def chat_loop():
    while True:
        user_input = input("\nPhil: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Build state
        state = {"messages": [HumanMessage(content=user_input)]}

        # 🔑 Invoke with thread_id so memory sticks
        result = agent.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        # ✅ Clean terminal output: only final AI response
        final_response = result["messages"][-1].content
        print("\nAI:", final_response, "\n")


if __name__ == "__main__":
    print("🤖 Type your research query (or 'quit' to exit).")
    chat_loop()
