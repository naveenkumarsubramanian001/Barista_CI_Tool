from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from datetime import datetime

# --- Agent Outputs ---

class DecomposedQueries(BaseModel):
    subqueries: List[str] = Field(..., description="List of 3-6 focused subqueries with time awareness.")

class Article(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str = Field(..., description="ISO format date YYYY-MM-DD")
    priority: bool = False

class SearchOutput(BaseModel):
    articles: List[Article]

class Insight(BaseModel):
    title: str
    brief_summary: str
    citation_id: int

class FinalReport(BaseModel):
    report_title: str
    top_insights: List[Insight]
    references: List[Article]

# --- Validation ---

class ValidationResult(BaseModel):
    approved: bool = Field(..., description="Whether the output is approved.")
    feedback: str = Field(..., description="Feedback/reason for rejection if not approved.")

# --- LangGraph State ---

from typing_extensions import TypedDict

class ResearchState(TypedDict):
    original_query: str
    subqueries: List[str]
    articles: List[Article]
    top_articles: List[Article]
    final_report: Optional[Dict]
    validation_feedback: str
    retry_counts: Dict[str, int]
    error: Optional[str]
