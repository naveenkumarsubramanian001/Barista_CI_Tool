import json
from langchain_core.prompts import ChatPromptTemplate
from config import get_llm
from utils.json_utils import safe_json_extract
from tavily import TavilyClient
from config import TAVILY_API_KEY
from models.schemas import AnalyzerState, ProductProfile, CompetitorProfile
from pydantic import BaseModel, Field
from typing import List

class CompetitorAnalysis(BaseModel):
    name: str
    domain: str
    strengths: List[str]
    weaknesses: List[str]
    pricing_strategy: str
    key_features: List[str]

class ComparativeReportModel(BaseModel):
    report_title: str
    executive_summary: str = Field(description="3-4 paragraphs summarizing the competitive landscape.")
    competitors: List[CompetitorAnalysis]
    user_product_positioning: str
    recommendations: List[str]
    
class DiscoveredCompetitor(BaseModel):
    name: str
    official_domain: str
    reason_for_inclusion: str

class DiscoveryOutput(BaseModel):
    competitors: List[DiscoveredCompetitor]

def product_extraction_agent(state: AnalyzerState) -> AnalyzerState:
    """Extracts features, value prop, target audience from uploaded text."""
    uploaded_text = state.get("uploaded_text", "")
    
    # Truncate if too huge to save tokens, though typical PDFs are fine
    text_to_process = uploaded_text[:50000]

    prompt = ChatPromptTemplate.from_template("""
    You are an expert product marketing analyst.
    Below is text extracted from a user's uploaded product document, brochure, or description.
    Read it carefully and extract the core product profile.
    
    Document Text:
    {text}
    
    You MUST return a valid JSON object matching this schema exactly:
    {{
      "product_name": "string",
      "features": ["string", "string"],
      "value_proposition": "string",
      "target_audience": "string",
      "market_positioning": "string"
    """)
    
    chain = prompt | get_llm().with_structured_output(ProductProfile)
    
    try:
        res = chain.invoke({"text": text_to_process})
        profile = res
        state["product_profile"] = profile.model_dump()
        state["logs"].append(f"✅ Extracted profile for user product: {profile.product_name}")
        state["workflow_status"] = "discovering"
        state["progress_percentage"] = 30
    except Exception as e:
        state["error"] = f"Extraction failed: {str(e)}"
        
    return state

def competitor_discovery_agent(state: AnalyzerState) -> AnalyzerState:
    """Discovers top 3 competitors based on product profile."""
    profile = state.get("product_profile")
    if not profile:
        return state
        
    prompt = ChatPromptTemplate.from_template("""
    You are a competitive intelligence bot.
    The user is building/owns this product:
    Name: {product_name}
    Features: {features}
    Value Prop: {value_proposition}
    Target Audience: {target_audience}
    Market: {market_positioning}
    
    Identify exactly 3 direct or strong indirect competitors in the real world.
    """)
    
    chain = prompt | get_llm().with_structured_output(DiscoveryOutput)
    
    try:
        res = chain.invoke(profile)
        profiles = [{"name": c.name, "official_domain": c.official_domain, "reason_for_inclusion": c.reason_for_inclusion} for c in res.competitors[:3]]
        state["discovered_competitors"] = profiles
        state["logs"].append(f"🔍 Discovered competitors: {', '.join([c['name'] for c in profiles])}")
        state["workflow_status"] = "researching"
        state["progress_percentage"] = 50
    except Exception as e:
        state["error"] = f"Competitor discovery failed: {str(e)}"
        
    return state

def competitor_research_agent(state: AnalyzerState) -> AnalyzerState:
    """Uses Tavily to fetch data on the discovered competitors."""
    competitors = state.get("discovered_competitors", [])
    if not competitors:
        return state
        
    competitor_data = {}
    
    tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
    
    for comp in competitors:
        domain = comp["official_domain"]
        query = f"{comp['name']} product features, pricing, and updates site:{domain}"
        state["logs"].append(f"🌐 Researching {comp['name']} ({domain})...")
        
        try:
            results = tavily_client.search(query=query, max_results=3, search_depth="advanced")
            articles = [{"title": r.get("title"), "url": r.get("url"), "snippet": r.get("content")} for r in results.get("results", [])]
            competitor_data[domain] = articles
        except Exception as e:
            competitor_data[domain] = []
            state["logs"].append(f"⚠️ Failed to research {comp['name']}: {str(e)}")
            
    state["competitor_data"] = competitor_data
    state["workflow_status"] = "summarising"
    state["progress_percentage"] = 75
    return state

def comparative_summariser_agent(state: AnalyzerState) -> AnalyzerState:
    """Takes the product profile and competitor data and generates the side-by-side JSON."""
    profile = state.get("product_profile")
    competitors = state.get("discovered_competitors", [])
    c_data = state.get("competitor_data", {})
    
    if not profile or not competitors:
        return state
        
    prompt = ChatPromptTemplate.from_template("""
    You are an elite competitive intelligence analyst.
    Generate a comprehensive comparative report comparing the User's Product to 3 Competitors based ONLY on the provided data.
    
    USER PRODUCT PROFILE:
    {profile}
    
    COMPETITORS DATA:
    {c_data_str}
    """)
    
    import json
    c_data_context = []
    for c in competitors:
        domain = c["official_domain"]
        articles = c_data.get(domain, [])
        c_data_context.append({
            "competitor_name": c["name"],
            "domain": domain,
            "research_snippets": articles
        })
        
    chain = prompt | get_llm().with_structured_output(ComparativeReportModel)
    
    try:
        res = chain.invoke({
            "profile": json.dumps(profile),
            "c_data_str": json.dumps(c_data_context, default=str)
        })
        
        report_data = res.model_dump()
        state["final_report"] = report_data
        state["logs"].append("✅ Comparative report generated successfully.")
        state["workflow_status"] = "completed"
        state["progress_percentage"] = 100
    except Exception as e:
        state["error"] = f"Summariser failed: {str(e)}"
        
    return state
