
# As part of learning, fix the issue with this langgraph flow so that
# it does not stop prematurely after calling knowledge graph.

from typing import Literal, TypedDict
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import InMemorySaver

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch
from firecrawl import Firecrawl

# ----------------- ENV -----------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
groq_base_url = os.getenv("GROQ_BASE_URL")
serp_api_key = os.getenv("SERP_API_KEY")
firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")

# ----------------- LLM -----------------
llm = ChatOpenAI(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=groq_api_key,
    base_url=groq_base_url,
    temperature=0.5,
)

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
    return {"knowledge_graph": knowledge_graph} if knowledge_graph else {"error": "No Knowledge Graph data found."}

@tool
def organic_google_search(query: str) -> dict:
    """Perform a general organic Google search. Returns top 6 results."""
    params = {"engine": "google", "q": query, "api_key": serp_api_key}
    search = GoogleSearch(params)
    results = search.get_dict()
    organic_results = results.get("organic_results", None)
    return {"results": organic_results[:6]} if organic_results else {"error": "No general search results found."}

@tool
def firecrawl_scrape(url: str) -> dict:
    """Scrape webpage content into a concise summary using Firecrawl."""
    try:
        doc = firecrawl.scrape(url, formats=["summary"])
        return {"url": url, "summary": getattr(doc, "summary", None), "metadata": getattr(doc, "metadata", None)}
    except Exception as e:
        return {"error": str(e)}

tools = [knowledge_graph_search, organic_google_search, firecrawl_scrape]
tools_by_name = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ----------------- EXTENDED STATE -----------------
class ExtendedState(MessagesState):
    candidate_url: str = ""
    organic_results: list = []
    needs_human_review: bool = False

# ----------------- HUMAN INTERVENTION NODE -----------------
def human_review_url(state: ExtendedState):
    """
    Pause for human to review organic search results and correct the URL if needed.
    Only called when needs_human_review is True.
    """
    if not state.get("needs_human_review", False):
        return {}
    
    print("\n--- Organic Search Top Results ---")
    organic_results = state.get("organic_results", [])
    
    if not organic_results:
        print("No organic results available for review.")
        return {"needs_human_review": False}
    
    for idx, res in enumerate(organic_results, start=1):
        print(f"{idx}. {res.get('link', 'No link')} - {res.get('title', 'No title')}")

    candidate_url = state.get("candidate_url", "")
    print(f"\nCandidate URL selected by AI: {candidate_url}")
    
    value = interrupt({"candidate_url": candidate_url})
    return {"candidate_url": value, "needs_human_review": False}

# ----------------- LLM NODE -----------------
def llm_call(state: ExtendedState):
    system_content = """
    You are a research assistant.

    Workflow:
    1. Try `knowledge_graph_search` first for the user's query.
    2. If knowledge graph data exists, use it directly to answer.
    3. If no knowledge graph data is found, perform `organic_google_search` for the same query.
    4. From organic search results, select the most relevant URL and use `firecrawl_scrape` to get detailed information.
    5. Finally, provide a comprehensive answer based on the scraped content.

    Remember: If one tool fails or returns no data, continue with the next step in the workflow.
    """
    
    messages = [SystemMessage(content=system_content)] + state["messages"]

    print("\n--- Intercepted Messages Before LLM ---")
    for msg in messages[-3:]:  # Show last 3 messages to avoid clutter
        print(f"[{msg.type.upper()}] {msg.content[:200]}{'...' if len(str(msg.content)) > 200 else ''}")
    print("--- End Intercept ---\n")

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ----------------- TOOL NODE -----------------
def tool_node(state: ExtendedState):
    results = []
    last_message = state["messages"][-1]
    updates = {}

    if getattr(last_message, "tool_calls", None):
        for tool_call in last_message.tool_calls:
            tool = tools_by_name[tool_call["name"]]
            print(f"\n--- Tool Call ---\nTool: {tool_call['name']}\nArgs: {tool_call['args']}\n----------------\n")
            observation = tool.invoke(tool_call["args"])
            results.append(ToolMessage(content=str(observation), tool_call_id=tool_call["id"]))
            print(f"--- Tool Result ---\n{observation}\n-------------------\n")

            # Handle organic search results - set up for human review
            if tool_call["name"] == "organic_google_search" and "results" in observation and observation["results"]:
                top_url = observation["results"][0]["link"] if observation["results"] else ""
                updates.update({
                    "candidate_url": top_url,
                    "organic_results": observation["results"],
                    "needs_human_review": True
                })

    updates["messages"] = results
    return updates

def should_continue(state: ExtendedState) -> Literal["Action", "human_review_url", END]:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "Action"
    return END

def should_review(state: ExtendedState) -> Literal["human_review_url", "environment", END]:
    """Decide whether human review is needed after tool execution."""
    if state.get("needs_human_review", False):
        return "human_review_url"
    
    # Check if we need to continue with scraping
    last_message = state["messages"][-1]
    if isinstance(last_message, ToolMessage):
        # If we just got organic search results, we might need to scrape
        # This is handled by the LLM deciding what to do next
        return END
    
    return END

# ----------------- GRAPH -----------------
graph_builder = StateGraph(ExtendedState)
graph_builder.add_node("llm_call", llm_call)
graph_builder.add_node("environment", tool_node)
graph_builder.add_node("human_review_url", human_review_url)

# Flow: START → LLM → Tool → [Human Review if needed] → END
graph_builder.add_edge(START, "llm_call")
graph_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
graph_builder.add_conditional_edges("environment", should_review, {
    "human_review_url": "human_review_url",
    "environment": "environment", 
    END: END
})
graph_builder.add_edge("human_review_url", END)

checkpointer = InMemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

# ----------------- INTERACTIVE LOOP WITH INTERRUPT -----------------
def chat_loop():
    while True:
        user_input = input("\nUser: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        state = {
            "messages": [HumanMessage(content=user_input)],
            "candidate_url": "",
            "organic_results": [],
            "needs_human_review": False
        }
        thread_id = str(os.urandom(4).hex())
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            result = graph.invoke(state, config=config)

            # Handle human interrupts
            while "__interrupt__" in result:
                interrupt_value = result["__interrupt__"][0].value
                print("\n⚠️ Human intervention needed!")
                print("Candidate URL selected by AI:", interrupt_value["candidate_url"])
                human_input = input("Enter correct URL if needed (or press enter to keep): ")
                resume_value = human_input if human_input.strip() else interrupt_value["candidate_url"]
                result = graph.invoke(Command(resume=resume_value), config=config)

            final_response = result.get("messages", [{"content": "No output"}])[-1].content
            print("Assistant:", final_response)
            
        except Exception as e:
            print(f"Error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    print("🤖 Research Assistant with Mid-Flow Human-in-the-Loop ready!")
    chat_loop()