from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from langchain.messages import SystemMessage, HumanMessage
from LLM.LLM_init import LLm_init
from  Tool.tool import rag_retrieve


# ------------------------
# App Init
# ------------------------
app = FastAPI(
    title="SHL Assessment Recommendation API",
    version="1.0.0"
)

# ------------------------
# LLM + Tool Init (load once)
# ------------------------
model = LLm_init()
model_with_tools = model.bind_tools([rag_retrieve])

system_msg = SystemMessage(
    content="""
You are an expert AI assistant specialized in recommending SHL (Saville and Holdsworth Limited) assessments.

Your Role:
- Analyze job requirements, skills, and competencies
- Recommend the most suitable SHL assessments

Rules:
- ALWAYS call the retrieval tool to fetch assessments
- DO NOT hallucinate or invent assessments
- Use ONLY verified data from the retrieval tool
- Return valid JSON with key 'recommendations'
- Each recommendation must include: id, name, url, test_type
- Limit recommendations to top 5 most relevant results
"""
)

# ------------------------
# Request / Response Schemas
# ------------------------
class RecommendRequest(BaseModel):
    query: str

class Assessment(BaseModel):
    id: str
    name: str
    url: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    query: str
    recommendations: List[Assessment]


# ------------------------
# Core Logic (unchanged)
# ------------------------
def run_query(user_query: str) -> Dict[str, Any]:
    messages = [
        system_msg,
        HumanMessage(content=user_query)
    ]

    response = model_with_tools.invoke(messages)

    # Tool call path (expected)
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        tool_output = rag_retrieve.invoke(tool_call["args"])

        return {
            "query": user_query,
            "recommendations": tool_output[:5]  # enforce max-5
        }

    # Fallback (should not happen)
    return {
        "query": user_query,
        "recommendations": []
    }


# ------------------------
# API Endpoints (SHL Spec)
# ------------------------
@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    result = run_query(req.query)

    if not result["recommendations"]:
        raise HTTPException(status_code=404, detail="No recommendations found")

    return result
