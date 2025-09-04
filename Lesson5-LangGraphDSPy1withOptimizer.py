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
from dspy.teleprompt import BootstrapFewShot

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
    max_tokens=200,   # Short responses for query optimization
    cache=True
)
dspy.configure(lm=groq_dspy_lm)

assess_lm = dspy.LM(
        model="groq/qwen/qwen3-32b",
        api_key=groq_api_key,
        temperature=0.2,
        cache=True
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


# Enhanced signature for query optimization
class SearchQueryOptimizationSig(dspy.Signature):
    """
    Transform raw search queries into optimized versions that improve search results.
    Focus on medical accuracy, searchability, and preserving critical details.
    """
    raw_query: str = dspy.InputField(
        desc="Original search query from the tool call - may be basic or incomplete"
    )
    optimized_query: str = dspy.OutputField(
        desc="Enhanced search query with improved terminology, relevant keywords, better phrasing, and preserved specificity - optimized for authoritative search results"
    )

trainset = [
    dspy.Example(
        raw_query="ephedrine safety for children under 1 year",
        optimized_query="ephedrine safety profile infants under 1 year pediatric contraindications dosage guidelines adverse effects"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="chlorpheniramine for children under 1 year",
        optimized_query="chlorpheniramine use in infants under 1 year safety warnings pediatric antihistamine dosage risks"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="fun fact giraffes",
        optimized_query="unique fun facts about giraffes including biology height feeding behavior social structure"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="python best libraries 2024",
        optimized_query="top Python libraries 2024 for data science machine learning web development productivity"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="climate change effects cities",
        optimized_query="climate change impacts on urban cities infrastructure flooding heatwaves adaptation strategies"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="AI tools productivity",
        optimized_query="best AI tools for productivity workflow automation task management knowledge work 2024"
    ).with_inputs("raw_query")
]

devset = [
    dspy.Example(
        raw_query="aspirin dosage children 2 years",
        optimized_query="aspirin dosage guidelines for children age 2 years pediatric dosing safety risks Reye syndrome"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="machine learning algorithms comparison",
        optimized_query="comparison of machine learning algorithms classification regression clustering performance metrics"
    ).with_inputs("raw_query"),
    
    dspy.Example(
        raw_query="sustainable energy solutions",
        optimized_query="sustainable renewable energy solutions including solar wind hydro storage implementation strategies"
    ).with_inputs("raw_query")
]

# LLM-assisted metric signature
class AssessQueryOptimization(dspy.Signature):
    """Assess the quality of an optimized search query."""
    original_query: str = dspy.InputField(desc="The original raw query before optimization")
    optimized_query: str = dspy.InputField(desc="The query that was optimized by the DSPy module")
    assessment_question: str = dspy.InputField(desc="The specific question to evaluate the optimized query")
    assessment_answer: bool = dspy.OutputField(desc="True if the optimized query meets the criteria, False otherwise")

def query_optimization_metric(example, pred, trace=None):
    """
    Evaluates if the predicted optimized query effectively improves searchability,
    clarity, and keyword coverage while preserving original intent.
    """
    raw_query = example.raw_query
    expected_optimized = example.optimized_query  # Gold standard
    
    # Handle case where pred might be a string or object
    if hasattr(pred, 'optimized_query'):
        predicted_optimized = pred.optimized_query
    elif isinstance(pred, str):
        predicted_optimized = pred
    else:
        # If pred is the full prediction object, extract the optimized_query
        predicted_optimized = getattr(pred, 'optimized_query', str(pred))

    # Metric 1: Searchability and Keyword Enhancement
    q1 = f"""
    Evaluate the searchability and keyword enhancement of the optimized query.
    
    Original Query: "{raw_query}"
    Predicted Optimized Query: "{predicted_optimized}"
    Expected Quality (Gold Standard): "{expected_optimized}"
    
    Does the predicted optimized query significantly improve searchability by:
    - Adding relevant domain-specific keywords?
    - Using better terminology for authoritative sources?
    - Including synonyms or related terms that expand search coverage?
    
    Compare the keyword richness and search effectiveness to the gold standard example.
    Respond with True if the predicted query has good searchability improvements, False otherwise.
    """
    
    # Metric 2: Clarity and Specificity Preservation
    q2 = f"""
    Evaluate whether the optimized query maintains clarity and preserves critical specificity.
    
    Original Query: "{raw_query}"
    Predicted Optimized Query: "{predicted_optimized}"
    Expected Quality (Gold Standard): "{expected_optimized}"
    
    Does the predicted optimized query:
    - Preserve all important details from the original (ages, numbers, specific requirements)?
    - Maintain or improve clarity of intent?
    - Avoid introducing ambiguity or changing the core meaning?
    
    Compare to the gold standard for maintaining specificity while enhancing searchability.
    Respond with True if specificity and clarity are well preserved, False otherwise.
    """
    
    # Metric 3: Professional Terminology and Context
    q3 = f"""
    Evaluate the appropriate use of professional terminology and context.
    
    Original Query: "{raw_query}"
    Predicted Optimized Query: "{predicted_optimized}"
    Expected Quality (Gold Standard): "{expected_optimized}"
    
    Does the predicted optimized query appropriately:
    - Add relevant professional or technical terms when beneficial?
    - Use domain-appropriate language (medical, technical, scientific)?
    - Enhance the query without over-complicating or adding irrelevant jargon?
    
    Compare the terminology appropriateness to the gold standard example.
    Respond with True if terminology is appropriately enhanced, False otherwise.
    """

    # Use the assessment LLM to evaluate each criterion
    with dspy.context(lm=assess_lm):
        searchability_eval = dspy.Predict(AssessQueryOptimization)(
            original_query=raw_query,
            optimized_query=predicted_optimized,
            assessment_question=q1
        )
        
        clarity_eval = dspy.Predict(AssessQueryOptimization)(
            original_query=raw_query,
            optimized_query=predicted_optimized,
            assessment_question=q2
        )
        
        terminology_eval = dspy.Predict(AssessQueryOptimization)(
            original_query=raw_query,
            optimized_query=predicted_optimized,
            assessment_question=q3
        )

    # Calculate score (0-3)
    score = (
        int(searchability_eval.assessment_answer) + 
        int(clarity_eval.assessment_answer) + 
        int(terminology_eval.assessment_answer)
    )
    
    # For debugging
    if trace is not None:
        print(f"Query Optimization Assessment:")
        print(f"  Searchability: {searchability_eval.assessment_answer}")
        print(f"  Clarity: {clarity_eval.assessment_answer}")
        print(f"  Terminology: {terminology_eval.assessment_answer}")
        print(f"  Total Score: {score}/3")

    # Return True if at least 2 out of 3 criteria are met
    return score >= 2

# Create and optimize the query optimizer module
class TrainedQueryOptimizer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(SearchQueryOptimizationSig)

    def forward(self, raw_query: str) -> str:
        result = self.predictor(raw_query=raw_query)
        return result.optimized_query

# ----------------- MODULE PERSISTENCE LOGIC -----------------
# Define the file path for saving the optimized module
OPTIMIZED_MODULE_FILE = "trained_query_optimizer.json"

# Global cache for the optimized module
trained_query_optimizer = None

def optimize_and_save_module():
    """
    Optimizes the TrainedQueryOptimizer using BootstrapFewShot,
    saves it, and returns the optimized module.
    """
    print("🔥 Starting DSPy query optimization training...")
    basic_optimizer = TrainedQueryOptimizer()

    # Configure BootstrapFewShot
    config = dict(max_labeled_demos=4)  # Use up to 4 training examples
    teleprompter = BootstrapFewShot(metric=query_optimization_metric, **config)

    # Compile (train) the module
    print("🔥 Training query optimizer with examples...")
    trained_module = teleprompter.compile(basic_optimizer, trainset=trainset)
    print("✅ Query optimizer training completed!")

    # Save the optimized module
    trained_module.save(OPTIMIZED_MODULE_FILE)
    print(f"💾 Optimized module saved to: {OPTIMIZED_MODULE_FILE}")

    # Evaluate on development set
    print("\n🔥 Evaluating trained optimizer on development set...")
    dev_scores = []
    for i, example in enumerate(devset):
        print(f"Evaluating example {i+1}/{len(devset)}...")
        prediction = trained_module(raw_query=example.raw_query)
        
        # Create prediction object for metric
        pred_obj = type('Prediction', (), {'optimized_query': prediction})()
        score = query_optimization_metric(example, pred_obj)
        dev_scores.append(int(score))
        
        print(f"  Original: '{example.raw_query}'")
        print(f"  Predicted: '{prediction}'")
        print(f"  Expected: '{example.optimized_query}'")
        print(f"  Score: {int(score)}/1\n")

    avg_dev_score = sum(dev_scores) / len(dev_scores)
    print(f"🎯 Average development score: {avg_dev_score:.2f}")

    return trained_module

def load_or_train_optimizer():
    """
    Loads the optimized module if it exists, otherwise trains and saves it.
    Returns the trained/loaded module.
    """
    global trained_query_optimizer
    
    # Check if module is already loaded in memory
    if trained_query_optimizer is not None:
        print("✅ Reusing cached DSPy module...")
        return trained_query_optimizer
    
    print("⚙️ Loading or training DSPy module...")
    
    # Check if saved module exists
    if os.path.exists(OPTIMIZED_MODULE_FILE):
        print(f"📦 Loading optimized module from {OPTIMIZED_MODULE_FILE}...")
        try:
            # Instantiate the module type *before* loading
            temp_module = TrainedQueryOptimizer()
            temp_module.load(OPTIMIZED_MODULE_FILE)
            trained_query_optimizer = temp_module
            print("✅ Module loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load module: {e}")
            print("🔄 Falling back to training new module...")
            trained_query_optimizer = optimize_and_save_module()
    else:
        print("📝 No saved module found. Training new module...")
        trained_query_optimizer = optimize_and_save_module()
    
    return trained_query_optimizer

# Initialize the optimizer (loads existing or trains new)
trained_query_optimizer = load_or_train_optimizer()

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

def enhanced_tool_node_with_training(state: dict):
    """Tool node using the trained DSPy query optimizer."""
    global trained_query_optimizer
    results = []
    
    # Since we now ensure only one tool call per batch, this should be simpler
    tool_calls = state["messages"][-1].tool_calls
    
    for tool_call in tool_calls:
        tool = tools_by_name[tool_call["name"]]
        
        # Safely extract args
        args = tool_call.get("args", {})
        if not isinstance(args, dict):
            args = {}
        
        # Use trained optimizer for query optimization
        if "query" in args and args["query"]:
            original_query = args["query"]
            
            try:
                # Use the trained optimizer once per tool call
                optimized_query = trained_query_optimizer(raw_query=original_query)
                args["query"] = optimized_query
                
                print(f"🔥 Trained DSPy Query Optimization:")
                print(f"   Tool: {tool_call['name']}")
                print(f"   Original: '{original_query}'")
                print(f"   Optimized: '{optimized_query}'")
                
            except Exception as e:
                print(f"⚠️ Trained optimization failed: {e}")
                print(f"🔄 Using original query: '{original_query}'")
        
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
agent_builder.add_node("environment", enhanced_tool_node_with_training)

agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"Action": "environment", END: END})
agent_builder.add_edge("environment", "llm_call")

# Attach memory
memory = InMemorySaver()
agent = agent_builder.compile(checkpointer=memory)

# Choose a thread id for this interactive run
THREAD_ID = "chat-session"

# ----------------- UTILITY FUNCTIONS -----------------
def reset_optimizer():
    """Force retrain the optimizer by deleting saved file and clearing cache."""
    global trained_query_optimizer
    
    if os.path.exists(OPTIMIZED_MODULE_FILE):
        os.remove(OPTIMIZED_MODULE_FILE)
        print(f"🗑️ Deleted saved module: {OPTIMIZED_MODULE_FILE}")
    
    trained_query_optimizer = None
    print("🔄 Module cache cleared. Next run will retrain the optimizer.")

def get_optimizer_status():
    """Check the status of the optimizer module."""
    global trained_query_optimizer
    
    file_exists = os.path.exists(OPTIMIZED_MODULE_FILE)
    module_cached = trained_query_optimizer is not None
    
    print(f"📊 Optimizer Status:")
    print(f"   Saved file exists: {file_exists}")
    print(f"   Module cached in memory: {module_cached}")
    print(f"   File path: {OPTIMIZED_MODULE_FILE}")

# ----------------- INTERACTIVE LOOP -----------------
def chat_loop():
    while True:
        user_input = input("\nPhil: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "reset":
            reset_optimizer()
            # Reload the optimizer
            load_or_train_optimizer()
            continue
        elif user_input.lower() == "status":
            get_optimizer_status()
            continue

        # Build state
        state = {"messages": [HumanMessage(content=user_input)]}

        # 🔒 Invoke with thread_id so memory sticks
        result = agent.invoke(
            state,
            config={"configurable": {"thread_id": THREAD_ID}},
        )

        # ✅ Clean terminal output: only final AI response
        final_response = result["messages"][-1].content
        print("\nAI:", final_response, "\n")


if __name__ == "__main__":
    print("🤖 Type your research query (or 'quit' to exit).")
    print("💡 Special commands: 'status' (check optimizer status), 'reset' (retrain optimizer)")
    chat_loop()