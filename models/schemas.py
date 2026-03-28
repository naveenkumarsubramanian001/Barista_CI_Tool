from typing import List, Optional, Dict
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

# --- Agent Outputs ---


class SubQuery(BaseModel):
    subquery: str = Field(..., description="A focused, specific search subquery")
    purpose: str = Field(..., description="What this subquery is trying to find")
    entity_focus: Optional[str] = Field(
        None, description="Primary entity this subquery targets"
    )


class DecomposedQueries(BaseModel):
    subqueries: List[SubQuery] = Field(
        ..., description="3-5 non-overlapping focused subqueries"
    )
    strategy: str = Field(
        ...,
        description="Overall decomposition strategy used (e.g. criteria-based, entity-based)",
    )


class CompanyCheck(BaseModel):
    is_company: bool


class CompanyList(BaseModel):
    companies: List[str] = Field(
        ..., description="List of validated company or organization names"
    )


class SuggestedCompanies(BaseModel):
    companies: List[str] = Field(
        ...,
        description="List of top 5 relevant company names for the given industry/context",
    )


class OfficialDomainSelection(BaseModel):
    official_url: Optional[str] = Field(
        None,
        description="The most likely official website URL from the candidates list",
    )
    is_official: bool = Field(
        ...,
        description="Whether the selected URL is truly an official company website or just a relevant page",
    )


class CategorySelection(BaseModel):
    category: str = Field(
        ...,
        description="The most relevant category for the query from the list of available categories.",
    )


class Article(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str = Field(..., description="ISO format date YYYY-MM-DD")
    source_type: str = Field(..., description="'official' or 'trusted'")
    domain: Optional[str] = Field(
        None, description="The domain the article was fetched from"
    )
    priority: bool = False
    score: float = Field(0.0, description="Quality score used for ranking/display")


class SearchOutput(BaseModel):
    articles: List[Article]


class Insight(BaseModel):
    title: str
    detailed_summary: str = Field(default="", description="A comprehensive, detailed summary of the article covering all facts, figures, and strategies.")
    reasoning: str = Field(default="", description="Explanation of how this article answers the original query")
    sentiment: str = Field(default="Neutral", description="Sentiment of the article toward the competitor")
    key_metrics: List[str] = Field(default_factory=list, description="Key metrics extracted")
    key_features: List[str] = Field(default_factory=list, description="Key features extracted")
    citation_id: int


class FinalReport(BaseModel):
    report_title: str
    executive_summary: str = Field(default="", description="A cohesive 2-3 paragraph synthesis answering the user's overarching query based on all articles.")
    conflict_and_consensus: str = Field(default="", description="Analysis comparing official vs trusted sources.")
    official_insights: List[Insight] = Field(default_factory=list)
    trusted_insights: List[Insight] = Field(default_factory=list)
    references: List[Article]


# --- Validation ---


class ValidationResult(BaseModel):
    approved: bool = Field(..., description="Whether the output is approved.")
    feedback: str = Field(
        ..., description="Feedback/reason for rejection if not approved."
    )


# --- LangGraph State ---


class ResearchState(TypedDict):
    original_query: str
    subqueries: List[str]
    official_sources: List[Dict]
    trusted_sources: List[Dict]
    final_ranked_output: Dict[str, List[Dict]]
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

    # Human-in-the-loop selection
    selected_articles: List[str]
    logs: List[str]

    # Control
    retry_counts: Dict[str, int]
    error: Optional[str]
    search_days_used: Optional[int]


# --- Analyzer Workflow State ---


class CompetitorProfile(BaseModel):
    name: str = Field(..., description="Name of the competitor")
    official_domain: str = Field(..., description="Official website domain of the competitor")
    reason_for_inclusion: str = Field(..., description="Why this competitor was selected")


class ProductProfile(BaseModel):
    product_name: str = Field(..., description="Name of the user's product/company")
    features: List[str] = Field(default_factory=list, description="Key features extracted")
    value_proposition: str = Field(..., description="Core value proposition")
    target_audience: str = Field(..., description="Target audience or ICP")
    market_positioning: str = Field(..., description="How it is positioned in the market")


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


# --- API Contract Models ---


class ApiError(BaseModel):
    code: str = Field(..., description="Stable machine-readable error code")
    message: str = Field(..., description="Human-readable error message")


class SearchStartResponse(BaseModel):
    api_version: str = Field(default="v1")
    session_id: str
    status: str
    error: Optional[ApiError] = None


class WorkflowStatusResponse(BaseModel):
    api_version: str = Field(default="v1")
    session_id: str
    status: str
    current_stage: str
    progress_percentage: int
    stages: List[Dict]
    logs: List[str]
    error: Optional[ApiError] = None


class AnalyzeStatusResponse(BaseModel):
    api_version: str = Field(default="v1")
    session_id: str
    status: str
    progress_percentage: int
    logs: List[str]
    report_data: Optional[Dict] = None
    error: Optional[ApiError] = None
