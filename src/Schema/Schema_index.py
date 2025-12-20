from typing import List
from pydantic import BaseModel, Field


class SHLAssessment(BaseModel):
    id: str = Field(..., description="Unique assessment identifier")
    name: str = Field(..., description="Assessment name")
    url: str = Field(..., description="SHL catalog URL")
    test_type: List[str] = Field(
        ..., description="Assessment test types (K, P, C, etc.)"
    )
    text_for_embedding: str = Field(
        ..., description="Cleaned text used for vector embedding"
    )
    
    
