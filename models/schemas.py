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
