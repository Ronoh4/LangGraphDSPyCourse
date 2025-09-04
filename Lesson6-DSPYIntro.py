# dspyintro: DSPy stands for Decralative Self-Improving Python, 
# an ML framework built around the idea that prompt engineering should be declarative, modular, and trainable

# Import modules/libraries
# you have to install dspy: pip install dspy==2.6.27 (for this course)
import dspy
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")

# Configure DSPy with preferred LLM (Claude Sonnet 4)
lm = dspy.LM('anthropic/claude-sonnet-4-20250514', api_key=anthropic_api_key, cache=True)
dspy.configure(lm=lm)

# Configure a separate LLM for the AI-assisted metric (e.g., Claude Sonnet)
assess_lm = dspy.LM('anthropic/claude-sonnet-4-20250514', api_key=anthropic_api_key, cache=True)

# Define the NEW Signature for Prompt Optimization
class PromptOptimizer(dspy.Signature):
    """
    Transforms a raw, generic, or vaguely defined prompt into a highly structured,
    explicit, and effective prompt designed for optimal performance by a Large Language Model (LLM).
    This process involves enhancing clarity, adding specific constraints, defining desired output formats,
    and providing illustrative examples to guide the LLM's response generation.
    """
    raw_prompt: str = dspy.InputField(
        desc="The initial, often loosely defined, or boilerplate-laden prompt with system and user parts. "
             "This prompt typically provides a general task description, basic persona, and minimal output expectations, "
             "similar to how a prompt might be initially structured in a framework like CrewAI before refinement."
    )
    improved_prompt: str = dspy.OutputField(
        desc="A meticulously crafted and expanded version of the raw prompt. "
             "This improved prompt has enhanced clarity and adds specific constraints like incorporates explicit instructions for the LLM's role, objectives, context, "
             "detailed task directives, precise requirements (e.g., length, content constraints), "
             "clear formatting guidelines, and relevant examples among others. "
             "The aim is to eliminate ambiguity and provide comprehensive guidance for generating a high-quality, "
             "predictable, and relevant output from the LLM."
    )

# Create a simple Predict module using the new signature
# This is the module that DSPy will optimize to perform the prompt improvement task
basic_module = dspy.Predict(PromptOptimizer)

# --- Create Train and Dev Sets for Prompt Improvement ---
trainset = [
    dspy.Example(
        raw_prompt=(
            "[System]: You are a Market Opportunity Explorer for AI in Personalized Fitness and Nutrition Coaching.\n"
            "A strategic thinker who collaborates with early-stage startups to uncover niche opportunities.\n"
            "Experienced in using market signals, customer behavior, and unmet pain points to guide innovation.\n\n"
            "Your personal goal is: Identify underserved needs, emerging demands, and untapped segments within "
            "the AI in Personalized Fitness and Nutrition Coaching domain\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Identify gaps and emerging opportunities in the "
            "AI in Personalized Fitness and Nutrition Coaching space using up-to-date 2025 data.\n\n"
            "This is the expected criteria for your final answer: A list of 8-10 market gaps or opportunities with 1-2 lines of context each.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        improved_prompt=(
            "[System]: ROLE: Market Opportunity Analyst\n"
            "DOMAIN: AI for Personalized Fitness and Nutrition Coaching\n"
            "OBJECTIVE: Identify strategic market gaps, emerging demands, and underserved segments.\n"
            "CONTEXT:\n"
            "- You are a domain expert who advises early-stage startups.\n"
            "- You use behavioral trends, market signals, and customer pain points.\n"
            "- Your output guides innovation and product strategy.\n"
            "[User]: TASK: Generate a list of 8-10 specific market opportunities in the domain of AI for Personalized Fitness and Nutrition Coaching.\n\n"
            "REQUIREMENTS:\n"
            "- Each opportunity must include:\n"
            "  • A clear, short title\n"
            "  • 1-2 lines of explanation grounded in user behavior, trends, or unmet needs\n"
            "- Avoid vague or overly general phrasing\n"
            "- Do not summarize or give analysis before or after the list\n\n"
            "FORMAT:\n"
            "1. [Opportunity Title] - Contextual explanation\n"
            "2. ...\n\n"
            "EXAMPLES:\n"
            "1. Personalized Menopause Coaching - A growing demand among middle-aged women for tailored fitness and nutrition plans that adapt to hormonal changes.\n"
            "2. AI Meal Planning for Gut Health - Rising interest in gut microbiome optimization creates a need for personalized meal recommendations based on digestive data.\n\n"
            "Begin your list now. Return only the structured output."
        )
    ).with_inputs("raw_prompt"),

    dspy.Example(
        raw_prompt=(
            "[System]: You are a Cybersecurity Consultant for Small Businesses. Your goal is to help small businesses "
            "with their cybersecurity needs. Your backstory: You're an expert who understands small business "
            "challenges and how to keep their data safe.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Advise a small business on common cybersecurity threats they face and how to protect themselves. "
            "Your expected output is a list of three threats and simple solutions.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        improved_prompt=(
            "[System]: **Consultant Profile:** Cybersecurity Specialist for Small & Medium Enterprises (SMEs).\n"
            "**Core Mission:** Provide concise, actionable security guidance tailored for businesses with limited IT resources.\n"
            "**Key Expertise:** Risk assessment, incident response planning, and practical security tool implementation.\n" 
            "**Target Audience:** Non-technical business owners and staff requiring clear, understandable advice.\n\n"
            "[User]: **Assignment:** Detail 4 common cybersecurity threats relevant to a typical small business, each with 2 practical mitigation steps.\n\n"
            "**Output Directives:**\n"
            "- Focus on immediately implementable and high-impact solutions.\n"
            "- Ensure advice is current and avoids jargon.\n"
            "- Present as a clear, numbered list.\n\n"
            "**Format Example:**\n"
            "1. **Threat:** [Threat Name]\n"
            "   **Mitigation:** [Actionable Step 1]; [Actionable Step 2]\n\n"
            "**Illustrative Entry:**\n"
            "1. **Threat:** Phishing and Spear Phishing Attacks\n"
            "   **Mitigation:** Conduct mandatory, regular employee training on identifying suspicious emails; Implement email gateway security with robust anti-phishing filters.\n\n"
            "Generate the cybersecurity threat guide now. Only the formatted list should be returned."
        )
    ).with_inputs("raw_prompt"),

    dspy.Example(
        raw_prompt=(
            "[System]: You are a Technology Risk Analyst whose goal is to look at new tech and find problems. "
            "Your backstory is that you are good at spotting risks in new digital systems and you specialize in IoT devices. \n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Identify security and privacy risks for deploying IoT in a smart home setting. "
            "Expected output is a list of 3 risks.\n\n"
            "This is the expected criteria for your final answer: Just a simple list of risks.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        improved_prompt=(
            "[System]:\n"
            "**Analyst Role:** IoT Security and Privacy Risk Assessment Specialist.\n"
            "**Primary Objective:** Systematically identify and articulate potential vulnerabilities in nascent IoT deployments.\n"
            "**Domain Expertise:** Comprehensive understanding of IoT device architecture, data flow, and emerging threat landscapes.\n"
            "**Guiding Ethos:** Proactive identification and concise articulation of risks to enable robust mitigation strategies.\n\n"
            "[User]:\n"
            "**Task Directive:** Compile a list of 3 critical security and privacy risks associated with the deployment of Internet of Things (IoT) devices within a typical smart home environment.\n\n"
            "**Content Specifications:**\n"
            "- Each risk must be clearly named and explained in 2 sentences, highlighting the potential impact.\n"
            "- Focus on risks inherent to consumer IoT devices and their interconnected nature.\n"
            "- Categorize risks for clarity (e.g., Data Security, Device Vulnerabilities, Privacy Concerns).\n\n"
            "**Expected Format:**\n"
            "## IoT Smart Home Risk Assessment\n"
            "\n"
            "### [Risk Category 1]\n"
            "- **[Risk Name 1]:** [Concise explanation of risk and impact]\n"
            "- **[Risk Name 2]:** [Concise explanation of risk and impact]\n"
            "\n"
            "### [Risk Category 2]\n"
            "- **[Risk Name 3]:** [Concise explanation of risk and impact]\n"
            "\n"
            "**Illustrative Entry:**\n"
            "## IoT Smart Home Risk Assessment\n"
            "\n"
            "### Device Vulnerabilities\n"
            "- **Weak Default Passwords:** Many IoT devices ship with easily guessable or hardcoded credentials, making them prime targets for unauthorized access and botnet recruitment.\n"
            "- **Unpatched Firmware:** Manufacturers often fail to provide timely security updates, leaving devices vulnerable to known exploits long after discovery.\n\n"
            "Generate the comprehensive risk assessment now. Provide only the markdown content, strictly adhering to the specified structure."
        )
    ).with_inputs("raw_prompt"),
]

devset = [
    dspy.Example(
        raw_prompt=(
            "[System]: You are a Climate Risk Analyst specializing in urban environments. Your expertise lies in using AI models "
            "to evaluate environmental risks and guide decision-making for city planners.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Use 2025 climate data to identify 3 climate-related risks "
            "that should influence the design of a coastal urban infrastructure project.\n\n"
            "This is the expected criteria for your final answer: A list of 5 specific climate risks with 1-2 lines of explanation each.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it!"
        ),
        improved_prompt=(
            "[System]:\n"
            "**Role:** Urban Climate Risk Analyst\n"
            "**Objective:** Surface actionable climate risks affecting city infrastructure design in coastal areas.\n"
            "**Domain:** AI-Driven Environmental Modeling\n\n"
            "[User]:\n"
            "**Task Directive:** List 5 critical climate risks that urban planners must consider when designing infrastructure for a coastal city.\n\n"
            "**Requirements:**\n"
            "- Each risk should be named clearly and briefly explained based on projected 2025 climate models.\n"
            "- Focus on risks relevant to construction, public safety, and city sustainability.\n"
            "- Do not include policy commentary or mitigation strategies.\n\n"
            "**Format:**\n"
            "1. [Risk Name] - [Short impact explanation]\n"
            "2. ...\n\n"
            "**Illustrative Entry:**\n"
            "1. Sea Level Rise - Coastal flooding events are expected to increase in frequency and severity due to rising tides and extreme weather.\n"
            "2. Heat Island Intensification - Dense urban areas will experience prolonged heat waves that strain public health and power systems.\n\n"
            "Generate your list of climate risks now. Only return the structured markdown list."
        )
    ).with_inputs("raw_prompt"),

    dspy.Example(
        raw_prompt=(
            "[System]: You are a Mental Health Resource Curator. Your goal is to find and share resources for people who need mental health support. "
            "Your backstory: You are good at finding reliable info and tools.\n\n"
            "To give my best complete final answer to the task respond using the exact following format:\n\n"
            "Thought: I now can give a great answer.\n\n"
            "[User]: Current Task: Provide resources for managing a specific mental health challenge. " # Generic task
            "Expected output: A list of resources.\n\n"
            "This is the expected criteria for your final answer: A list of resources for the specified challenge.\n\n"
            "You MUST return the actual complete content as the final answer, not a summary.\n\n"
            "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
        ),
        improved_prompt=(
            "[System]:\n"
            "ROLE: Mental Wellness Resource Specialist\n"
            "DOMAIN: Mental Health Support and Stress Management\n"
            "OBJECTIVE: Curate and present accessible, evidence-based resources to improve mental well-being.\n"
            "CONTEXT:\n"
            "- You specialize in identifying credible tools, techniques, and programs for laypeople.\n"
            "- Your audience includes individuals seeking practical self-help options.\n"
            "- Resources should be current, actionable, and diverse in approach.\n\n"
            "[User]: TASK: List 4 high-quality resources or strategies for managing stress, tailored to the general public.\n\n"
            "REQUIREMENTS:\n"
            "- Each entry must include a name and a brief description (1-2 sentences).\n"
            "- Include at least one digital tool (e.g., app or website) and one offline method.\n"
            "- Ensure descriptions highlight how the resource helps with stress.\n\n"
            "FORMAT:\n"
            "- **Resource 1:** [Name] - [Description]\n"
            "- **Resource 2:** [Name] - [Description]\n"
            "- ...\n\n"
            "SAMPLE ENTRY:\n"
            "- **Resource 1:** Calm App - This mobile app offers guided meditations and breathing exercises to reduce stress and promote relaxation.\n"
            "- **Resource 2:** Journaling - Writing daily thoughts and feelings helps process emotions, lowering stress levels over time.\n\n"
            "Generate the resource list now. Provide only the formatted output."
        )
    ).with_inputs("raw_prompt"),

]

# --- Define an AI-Assisted Metric ---
class AssessPromptImprovement(dspy.Signature):
    """Assess the quality of an improved prompt."""
    assessed_raw_prompt: str = dspy.InputField(desc="The original raw prompt before improvement.")
    assessed_improved_prompt: str = dspy.InputField(desc="The prompt that was generated by the DSPy module based on the raw prompt.")
    assessment_question: str = dspy.InputField(desc="The specific question to evaluate the improved prompt.")
    assessment_answer: bool = dspy.OutputField(desc="True if the improved prompt meets the criteria, False otherwise.")


def prompt_improvement_metric(example, pred, trace=None):
    """
    Evaluates if the predicted improved prompt (pred.improved_prompt) effectively transforms
    the raw prompt (example.raw_prompt) into a high-quality, actionable, and well-structured prompt,
    consistent with the standards set by the example.improved_prompt.
    """
    raw_prompt = example.raw_prompt
    expected_improved_prompt = example.improved_prompt # This is the gold standard improved prompt
    predicted_improved_prompt = pred.improved_prompt # This is what our DSPy module generated

    # Metric 1: Structural Integrity & Clarity
    # Does the generated improved prompt follow a clear, well-structured format
    # and is it easy to parse for the LLM, similar to the golden examples?
    q1 = f"""
    Evaluate the structural integrity and clarity of the 'predicted_improved_prompt'.
    
    Predicted Improved Prompt:
    ---
    {predicted_improved_prompt}
    ---
    
    Compared to the high-quality format and clarity of the 'expected_improved_prompt':
    ---
    {expected_improved_prompt}
    ---
    
    Is the 'predicted_improved_prompt' well-structured, easy to parse, and clear in its instructions, using effective formatting (e.g., bolding, bullet points, clear sections) similar to the 'expected_improved_prompt'?
    Respond with True or False.
    """
    
    # Metric 2: Instructional Completeness & Specificity
    # Does the generated improved prompt fully elaborate on the generic requirements
    # from the raw prompt by adding specific constraints, examples, or detailed instructions,
    # making it more effective for the LLM to generate the desired output,
    # as demonstrated by the golden examples?
    q2 = f"""
    Evaluate the instructional completeness and specificity of the 'predicted_improved_prompt'.
    
    Raw Prompt:
    ---
    {raw_prompt}
    ---
    
    Predicted Improved Prompt:
    ---
    {predicted_improved_prompt}
    ---
    
    Compared to the comprehensive and specific instructions of the 'expected_improved_prompt':
    ---
    {expected_improved_prompt}
    ---
    
    Does the 'predicted_improved_prompt' sufficiently elaborate on the generic requirements from the 'raw_prompt' by adding specific constraints, clear formatting guidance, and concrete examples to ensure the LLM generates the *desired final output* effectively, mirroring the thoroughness of the 'expected_improved_prompt'?
    Respond with True or False.
    """

    with dspy.context(lm=assess_lm):
        clarity_eval = dspy.Predict(AssessPromptImprovement)(
            assessed_raw_prompt=raw_prompt, # Pass raw_prompt for context
            assessed_improved_prompt=predicted_improved_prompt,
            assessment_question=q1
        )
        completeness_eval = dspy.Predict(AssessPromptImprovement)(
            assessed_raw_prompt=raw_prompt, # Pass raw_prompt for context
            assessed_improved_prompt=predicted_improved_prompt,
            assessment_question=q2
        )

    score = int(clarity_eval.assessment_answer) + int(completeness_eval.assessment_answer)

    # For BootstrapFewShot, the metric typically returns True/False for each example
    # indicating if it's a good example to learn from.
    # Here, we can return True if both criteria are met.
    if trace is not None:
        # trace here would typically be for debugging, not the score itself.
        # For trace, you might want to print the individual assessment answers.
        pass

    return score == 2 # Return True if both criteria are met, False otherwise
       
# Import BootstrapFewShot
from dspy.teleprompt import BootstrapFewShot

config = dict(max_labeled_demos=3)  # Limit to 3 labeled demos for bootstrapping

# BootstrapFewShot will use the provided metric to guide its optimization
teleprompter = BootstrapFewShot(metric=prompt_improvement_metric, **config)

# --- Compile (Optimize) the Module ---
# This is where DSPy learns to transform raw_prompt into improved_prompt
print("Starting module optimization with bootstrapping...")
optimized_module = teleprompter.compile(basic_module, trainset=trainset)
print("Module optimized successfully!")

print("\n--- Evaluating on Development Set (Manual Loop) ---")
dev_scores = []
for i, example in enumerate(devset):
    print(f"Evaluating example {i+1}/{len(devset)}...")
    prediction = optimized_module(raw_prompt=example.raw_prompt) # Corrected input field
    score = prompt_improvement_metric(example, prediction) # Corrected metric function
    dev_scores.append(score)

avg_dev_score = sum(dev_scores) / len(dev_scores)
print(f"\nAverage score on the development set: {avg_dev_score:.2f}")

new_raw_prompt = (
    "[System]: You are a Sustainability Consultant specializing in corporate environmental impact. Your goal is to advise "
    "companies on reducing their carbon footprint and improving resource efficiency. Your backstory: You are an expert "
    "in analyzing supply chains, energy consumption, and waste management practices.\n\n"
    "Your personal goal is: Identify actionable strategies and innovative technologies for businesses to achieve "
    "net-zero emissions and circular economy principles by 2030.\n\n"
    "To give my best complete final answer to the task respond using the exact following format:\n\n"
    "Thought: I now can give a great answer.\n\n"
    "[User]: Current Task: Propose 4-5 actionable sustainability initiatives for a mid-sized manufacturing company, "
    "focusing on immediate impact and long-term viability, considering current industry trends (2025).\n\n"
    "This is the expected criteria for your final answer: A bulleted list of initiatives, each with a brief description "
    "and anticipated benefit (1-2 sentences per initiative).\n\n"
    "You MUST return the actual complete content as the final answer, not a summary.\n\n"
    "Begin! This is VERY important to you, use the tools available and give your best Final Answer, your job depends on it! "
)

# Use the optimized module to generate an improved prompt
result = optimized_module(raw_prompt=new_raw_prompt)

print("\n--- Optimized Prompt Generated for New Sustainability Example ---")
print(result.improved_prompt)

# Display the last prompt(s) used by the optimized module (optional, can show multiple if n > 1)
print("\n--- Prompt History for Second Final Prediction ---")
dspy.inspect_history(n=1) # Show only the last prompt used by this new call