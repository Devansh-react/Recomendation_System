
from fastapi.middleware.cors import CORSMiddleware

from langchain.tools import tool
from langchain.messages import SystemMessage, HumanMessage
from src.LLM.LLM_init import LLm_init
from  src.Tool.tool import rag_retrieve
# ------------------------
# LLM + Tool Init (load once)
# ------------------------

# For demonstration, let's define a hypothetical second tool.
# In a real project, this would likely be in its own file.
@tool
def check_pricing(assessment_name: str) -> str:
    """Useful for looking up the price of a specific SHL assessment when a user asks for cost or pricing."""
    # In a real app, this would query a database or another API.
    if "swift" in assessment_name.lower():
        return "The Swift Analysis Aptitude test costs $150."
    else:
        return f"Pricing for '{assessment_name}' is not available at the moment."

model = LLm_init()

# 1. Bind ALL tools to the model. The LLM will use their docstrings to decide which one to call.
tools = [rag_retrieve, check_pricing]
model_with_tools = model.bind_tools(tools)

# 2. Create a mapping from the tool's string name to its actual function. This is key for routing.
tool_map = {t.name: t for t in tools}

system_msg = SystemMessage(
    content="""
        You are an expert AI assistant for SHL (Saville and Holdsworth Limited).

        Your Role:
        - Analyze job requirements to recommend suitable SHL assessments.
        - Look up the price of a specific assessment if asked.

        Rules:
        - ALWAYS call a tool to answer the user. Use 'rag_retrieve' for recommendations and 'check_pricing' for prices.
        - DO NOT hallucinate or invent assessments or prices.
        - Use ONLY verified data from the tools.
        - Return valid JSON with key 'recommendations'
        - Each recommendation must include: id, name, url, test_type
        - Limit recommendations to top 5 most relevant results """
        )


def process_recommendation(user_query: str, messages):
    """Process user query and return recommendations."""
    response = model_with_tools.invoke(messages)

    # 3. The LLM's response contains the name of the tool it decided to use.
    if response.tool_calls:
        # The model can call multiple tools, so we loop through them.
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # 4. Use the map to find and run the correct tool function with the arguments from the LLM.
            if tool_name in tool_map:
                tool_to_run = tool_map[tool_name]
                tool_output = tool_to_run.invoke(tool_args)

                # --- Handle the output based on which tool was called ---
                # This part is specific to your application's needs.
                if tool_name == "rag_retrieve":
                    return {
                        "query": user_query,
                        "recommendations": tool_output[:5]  # enforce max-5
                    }
                elif tool_name == "check_pricing":
                    # We adapt the string output of the pricing tool to fit the API's response schema.
                    return {
                        "query": user_query,
                        "recommendations": [{"id": "price_info", "name": tool_output, "url": "", "test_type": []}]
                    }

    # Fallback if no tool was called or the tool name is not in our map
    return {
        "query": user_query,
        "recommendations": []
    }


