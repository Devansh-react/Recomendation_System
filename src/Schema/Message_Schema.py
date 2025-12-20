from typing import List
from pydantic import BaseModel, Field

class Assessment(BaseModel):
    id: str = Field(..., description="Unique assessment identifier")
    name: str = Field(..., description="Assessment name")
    url: str = Field(..., description="SHL catalog URL")
    test_type: List[str] = Field(
        ..., description="Assessment test types (K, P, C, etc.)"
    )


class RecommendationResponse(BaseModel):
    recommendations: List[Assessment]
