from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

# --- Agent Outputs ---

class SubQuery(BaseModel):
    subquery: str = Field(..., description="A focused, specific search subquery")
    purpose: str = Field(..., description="What this subquery is trying to find")
    entity_focus: Optional[str] = Field(None, description="Primary entity this subquery targets")

class DecomposedQueries(BaseModel):
    subqueries: List[SubQuery] = Field(..., description="3-5 non-overlapping focused subqueries")
    strategy: str = Field(..., description="Overall decomposition strategy used (e.g. criteria-based, entity-based)")

class CompanyCheck(BaseModel):
    is_company: bool

class CompanyList(BaseModel):
    companies: List[str] = Field(..., description="List of validated company or organization names")

class SuggestedCompanies(BaseModel):
    companies: List[str] = Field(..., description="List of top 5 relevant company names for the given industry/context")

class OfficialDomainSelection(BaseModel):
    official_url: Optional[str] = Field(None, description="The most likely official website URL from the candidates list")
    is_official: bool = Field(..., description="Whether the selected URL is truly an official company website or just a relevant page")

class CategorySelection(BaseModel):
    category: str = Field(..., description="The most relevant category for the query from the list of available categories.")


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
    company_domains: List[str]
    trusted_domains: List[str]

    # Validation
    validation_feedback: str
    validation_passed: bool
    validation_metrics: Dict[str, float]
    decomposition_score: float
    redundancy_pairs: List[List[str]]
    coverage_gaps: List[str]
    semantic_warnings: List[str]

    # Control
    retry_counts: Dict[str, int]
    error: Optional[str]