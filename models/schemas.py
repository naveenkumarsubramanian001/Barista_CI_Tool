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
    source_type: str = Field(..., description="'official' or 'trusted'")
    domain: Optional[str] = Field(None, description="The domain the article was fetched from")
    priority: bool = False
    score: float = Field(0.0, description="Quality score from hybrid fuzzy discriminator")

class SearchOutput(BaseModel):
    articles: List[Article]

class Insight(BaseModel):
    title: str
    brief_summary: str
    citation_id: int

class KeyFinding(BaseModel):
    finding_title: str = Field(..., description="Short title for the thematic finding")
    finding_summary: str = Field(..., description="3-5 sentence cross-source analysis of this finding")
    source_ids: List[int] = Field(default_factory=list, description="Citation IDs backing this finding")

class FinalReport(BaseModel):
    report_title: str
    executive_summary: str = Field("", description="3-4 paragraph overview synthesizing ALL sources")
    key_findings: List[KeyFinding] = Field(default_factory=list, description="Cross-source thematic findings")
    official_insights: List[Insight] = Field(default_factory=list)
    trusted_insights: List[Insight] = Field(default_factory=list)
    cross_source_analysis: str = Field("", description="Comparing official vs trusted perspectives")
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
    official_sources: List[Article]
    trusted_sources: List[Article]
    final_ranked_output: Dict[str, List[Article]]
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
    search_days_used: Optional[int]
    selected_articles: List[str]
    logs: List[str]

# --- Analyzer Workflow State ---

class CompetitorProfile(BaseModel):
    name: str = Field(..., description="Name of the competitor")
    official_domain: str = Field(..., description="Official website domain of the competitor")
    reason_for_inclusion: str = Field(..., description="Why this competitor was selected")

class ProductProfile(BaseModel):
    product_name: str = Field(..., description="Name of the user's product/company")
    features: List[str] = Field(default_factory=list, description="Key features extracted")
    value_proposition: str = Field(..., description="Core value prop")
    target_audience: str = Field(..., description="Target audience or ICP")
    market_positioning: str = Field(..., description="How it positions itself in the market")

class AnalyzerState(TypedDict):
    session_id: str
    uploaded_text: str
    product_profile: Optional[Dict]
    discovered_competitors: List[Dict]
    competitor_data: Dict[str, List[Dict]]
    final_report: Optional[Dict]
    logs: List[str]
    workflow_status: str
    progress_percentage: int
    error: Optional[str]
