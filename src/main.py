from langchain.messages import SystemMessage, HumanMessage

from LLM.LLM_init import LLm_init
from Tool.tool import rag_retrieve

# Init model
model = LLm_init()

# Bind tools
model_with_tools = model.bind_tools([rag_retrieve])

system_msg = SystemMessage(
    content="""
You are an expert AI assistant specialized in recommending SHL (Saville and Holdsworth Limited) assessments for recruitment and talent development.

Your Role:
- Analyze job requirements, skills, and competencies to identify the most suitable SHL assessments
- Provide tailored assessment recommendations based on specific hiring needs or talent evaluation scenarios
- Ensure recommendations are evidence-based and aligned with organizational goals

Rules:
- ALWAYS call the retrieval tool to fetch assessments - do not bypass this step
- DO NOT hallucinate, invent, or suggest assessments not in the tool output
- Use ONLY the verified data returned by the retrieval tool
- Return a valid JSON object with key "recommendations"
- Each recommendation must include: id, name, url, test_type
- Prioritize assessments that best match the candidate profile and role requirements
- Limit recommendations to the top 5 most relevant results

Response Format:
- Be concise and professional
- Explain why each assessment is recommended when appropriate
- Consider the candidate's seniority level, role type, and skill requirements
"""
)


def run_query(user_query: str):
    messages = [
        system_msg,
        HumanMessage(content=user_query)
    ]

    response = model_with_tools.invoke(messages)

    # If model decides to call tool
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_output = rag_retrieve.invoke(tool_call["args"])

        # Enforce final structure in Python (not LLM)
        final_response = {
            "query": user_query,
            "recommendations": tool_output[:5]  # enforce max-10
        }

        return final_response

    # Fallback (should not normally happen)
    return {
        "query": user_query,
        "recommendations": []
    }


if __name__ == "__main__":
    result = run_query(
        "Hiring a Java developer with strong collaboration skills"
    )
    print(result)
