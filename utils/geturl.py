import asyncio
import tldextract
import os
from typing import List, Optional
from pydantic import BaseModel, Field
from tavily import TavilyClient
from models.schemas import ResearchState, CompanyCheck, CompanyList, SuggestedCompanies, OfficialDomainSelection
from config import TAVILY_API_KEY, get_llm
from utils.predefinedurls import detect_category, get_domains_by_category
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM
llm = get_llm()

# --- Batch Validation Chain ---
validation_parser = JsonOutputParser(pydantic_object=CompanyList)
validation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an entity classification system. Filter the provided list and return ONLY the items that are actual companies, organizations, or brands."),
    ("user", "Entities: {entities}\n\n{format_instructions}")
])
validation_chain = validation_prompt | llm | validation_parser

# --- Dynamic Suggestion Chain ---
suggestion_parser = JsonOutputParser(pydantic_object=SuggestedCompanies)
suggestion_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an entity extraction expert. Based on the user's query, identify ONLY the primary companies, brands, or organizations explicitly mentioned or directly targeted by the query.\nCRITICAL RULES:\n1. If the user mentions a specific company (e.g., 'OpenAI', 'Anthropic'), ONLY return those.\n2. DO NOT list competitors or unrelated market leaders unless explicitly requested.\n3. Use the search context only to resolve partial names to their official company names.\n4. Return an empty list if no specific company is targeted.\nEnsure the generated output is a valid JSON with a 'companies' array containing string values."),
    ("user", "Search Context: {context}\n\nQuery: {query}\n\n{format_instructions}")
])
suggestion_chain = suggestion_prompt | llm | suggestion_parser

# --- Official Domain Selection Chain ---
selection_parser = JsonOutputParser(pydantic_object=OfficialDomainSelection)
selection_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a brand verification expert. From the provided list of search results, identify the ONE URL that is the primary, official homepage of the company. Official sites include localized versions (e.g., /in, /global, /en) and regional subdomains. Avoid support pages, news articles, or social media if a main homepage is available. If no result looks like an official company website, return is_official=False and official_url=None."),
    ("user", "Company: {company}\nCandidates:\n{candidates}\n\n{format_instructions}")
])
selection_chain = selection_prompt | llm | selection_parser

# --- Primary Entity Extraction Chain ---
# Identifies the SINGLE primary company a query is explicitly focused ON.
# Uses no external search context — query text only — to prevent competitor
# names from leaking into the official-domain list.
class PrimaryEntity(BaseModel):
    primary_company: Optional[str] = Field(
        None,
        description="The single primary company/brand the query is focused ON, or null if no single company is clearly targeted"
    )

primary_entity_parser = JsonOutputParser(pydantic_object=PrimaryEntity)
primary_entity_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an entity extraction expert.\n"
        "Given a user query, identify the SINGLE primary company or brand that the query is ABOUT.\n"
        "STRICT RULES:\n"
        "1. Only return a company if the query is clearly and explicitly focused on ONE specific company.\n"
        "2. 'Anthropic competitors' → primary_company: 'Anthropic' (the user is researching Anthropic)\n"
        "3. 'OpenAI latest products' → primary_company: 'OpenAI'\n"
        "4. 'best AI companies 2025' → primary_company: null (no single primary company)\n"
        "5. 'iPhone vs Android' → primary_company: null (multiple brands, no clear single primary)\n"
        "6. DO NOT return competitors, market leaders, or companies inferred from context.\n"
        "7. Return null if you are not highly confident a single primary company is targeted.",
    ),
    ("user", "Query: {query}\n\n{format_instructions}")
])
primary_entity_chain = primary_entity_prompt | llm | primary_entity_parser

async def validate_companies_batch(entities: List[str]) -> List[str]:
    """Validates a list of entities as companies in a single LLM call."""
    if not entities:
        return []
    try:
        result = await validation_chain.ainvoke({
            "entities": ", ".join(entities),
            "format_instructions": validation_parser.get_format_instructions()
        })
        return result.get("companies", [])
    except Exception as e:
        print(f"   - Error in batch validation for {entities}: {e}")
        return []

async def suggest_companies_dynamic(query: str) -> List[str]:
    """Suggests relevant companies for a generic query using real-time search and LLM."""
    context = ""
    
    # Try to get search context (Tavily first, then Google, then none)
    try:
        if TAVILY_API_KEY:
            client = TavilyClient(api_key=TAVILY_API_KEY)
            search_results = await asyncio.to_thread(
                client.search, 
                query=f"official company website or companies related to: {query}", 
                max_results=5
            )
            results_list = search_results.get('results', [])
            context = "\n".join([f"Result: {r.get('content', '')}" for r in results_list])
            print(f"   - Tavily context search found {len(results_list)} results.")
    except Exception as e:
        print(f"   - Tavily context search failed: {e}")
    
    # Fallback: try Google for context
    if not context:
        try:
            import json
            import urllib.request
            import urllib.parse
            google_key = os.getenv("GOOGLE_API_KEY", "").strip()
            google_cx = os.getenv("GOOGLE_CSE_ID", "").strip()
            if google_key and google_cx:
                params = urllib.parse.urlencode({
                    "key": google_key, "cx": google_cx,
                    "q": f"{query} official company", "num": 5
                })
                url = f"https://www.googleapis.com/customsearch/v1?{params}"
                req = urllib.request.Request(url, headers={"Accept": "application/json"})
                raw = await asyncio.to_thread(
                    lambda: urllib.request.urlopen(req, timeout=15).read().decode("utf-8")
                )
                data = json.loads(raw)
                context = "\n".join([f"Result: {i.get('snippet', '')}" for i in data.get("items", [])])
                print(f"   - Google context search found {len(data.get('items', []))} results.")
        except Exception as e:
            print(f"   - Google context search failed: {e}")
    
    # If still no context, use a direct LLM extraction (no context needed)
    if not context:
        context = f"Query: {query}. No external search results available. Extract companies directly from the query text."
        print(f"   - Using LLM-only extraction (no search context available).")

    try:
        result = await suggestion_chain.ainvoke({
            "query": query,
            "context": context,
            "format_instructions": suggestion_parser.get_format_instructions()
        })
        companies = result.get("companies", [])
        
        # Robust parsing fallback
        if not companies:
            for k, v in result.items():
                if isinstance(v, list) and len(v) > 0:
                    companies = v
                    break
                    
        print(f"   - LLM extracted {len(companies)} companies from context.")
        return companies
    except Exception as e:
        print(f"   - Error in dynamic suggestion for '{query}': {e}")
        return []

async def extract_primary_entity(query: str) -> Optional[str]:
    """
    Identifies the single primary company the query is explicitly focused ON.

    This uses only the raw query text (no external search context) to prevent
    competitor company names from contaminating the official-domain list.

    Examples:
      'Anthropic competitors'  → 'Anthropic'
      'OpenAI latest products' → 'OpenAI'
      'best AI companies'      → None

    Returns the company name string, or None if no single primary entity is found.
    """
    try:
        result = await primary_entity_chain.ainvoke({
            "query": query,
            "format_instructions": primary_entity_parser.get_format_instructions()
        })
        primary = result.get("primary_company")
        if primary and isinstance(primary, str) and primary.strip():
            print(f"   - Primary entity extracted: '{primary.strip()}'")
            return primary.strip()
        print("   - No single primary entity found in query.")
        return None
    except Exception as e:
        print(f"   - Primary entity extraction failed: {e}")
        return None


async def extract_companies(query: str, entities: List[str] = None) -> List[str]:
    """
    Extracts and validates the primary company entities from a query.

    Resolution order:
      1. Try to identify a single primary entity from the query text (no search
         context). This prevents competitor names from leaking into the
         official-domain list for queries like 'Anthropic competitors'.
      2. If specific NER entities are provided, validate them as companies.
      3. If still nothing, fall back to dynamic suggestion via web search + LLM.
    """
    companies = []

    # 1. Primary entity extraction — query-text only, no external context.
    #    For queries like 'Anthropic competitors' this returns ['Anthropic'] only,
    #    ensuring openai.com is never added as an official domain.
    primary = await extract_primary_entity(query)
    if primary:
        validated = await validate_companies_batch([primary])
        if validated:
            print(f"   - Using primary entity: {validated}")
            return list(dict.fromkeys(validated))[:5]

    # 2. Batch validate NER entities if present (fallback)
    if entities:
        companies = await validate_companies_batch(entities)

    # 3. Dynamic Suggestion if no companies identified yet
    if not companies:
        print("   - Using LLM to dynamically suggest relevant companies...")
        companies = await suggest_companies_dynamic(query)

    # Deduplicate and limit
    return list(dict.fromkeys(companies))[:5]

async def find_single_official_domain(company: str, client) -> str:
    """Try Tavily to find domain for a single company."""
    if not client:
        return None
    try:
        search_query = f"{company} official website"
        response = await asyncio.to_thread(client.search, query=search_query, max_results=3)
        results = response.get('results', [])
        if not results:
            return None

        candidate_text = "\n".join([f"- {r.get('title')}: {r.get('url')}" for r in results])
        selection = await selection_chain.ainvoke({
            "company": company,
            "candidates": candidate_text,
            "format_instructions": selection_parser.get_format_instructions()
        })
        
        official_url = selection.get("official_url")
        if not selection.get("is_official") or not official_url:
            return None

        ext = tldextract.extract(official_url)
        if ext.domain and ext.suffix:
            root_domain = f"{ext.domain}.{ext.suffix}"
            noise_domains = ["youtube", "wikipedia", "linkedin", "facebook", "twitter", "amazon"]
            if ext.domain not in noise_domains:
                return root_domain
    except Exception as e:
        print(f"   - Tavily domain lookup failed for {company}: {e}")
    return None


async def find_single_official_domain_google(company: str) -> str:
    """Fallback: Use Google Custom Search to find the company domain."""
    import json
    import urllib.request
    import urllib.parse
    google_key = os.getenv("GOOGLE_API_KEY", "").strip()
    google_cx = os.getenv("GOOGLE_CSE_ID", "").strip()
    if not google_key or not google_cx:
        return None
    try:
        params = urllib.parse.urlencode({
            "key": google_key, "cx": google_cx,
            "q": f"{company} official website", "num": 3
        })
        url = f"https://www.googleapis.com/customsearch/v1?{params}"
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        
        for item in data.get("items", []):
            link = item.get("link", "")
            ext = tldextract.extract(link)
            if ext.domain and ext.suffix:
                root = f"{ext.domain}.{ext.suffix}"
                noise = ["youtube", "wikipedia", "linkedin", "facebook", "twitter", "amazon"]
                if ext.domain not in noise:
                    # Let LLM verify it's official
                    candidate_text = "\n".join([f"- {i.get('title')}: {i.get('link')}" for i in data.get("items", [])])
                    try:
                        selection = await selection_chain.ainvoke({
                            "company": company,
                            "candidates": candidate_text,
                            "format_instructions": selection_parser.get_format_instructions()
                        })
                        if selection.get("is_official") and selection.get("official_url"):
                            ext2 = tldextract.extract(selection["official_url"])
                            if ext2.domain and ext2.suffix:
                                return f"{ext2.domain}.{ext2.suffix}"
                    except:
                        pass
                    return root
        return None
    except Exception as e:
        print(f"   - Google domain lookup failed for {company}: {e}")
        return None


async def find_single_official_domain_serper(company: str) -> str:
    """Fallback: Use Serper to find the company domain."""
    import json
    import urllib.request
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key:
        return None
    try:
        data = json.dumps({"q": f"{company} official website", "num": 3}).encode("utf-8")
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=data,
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            results = json.loads(resp.read().decode("utf-8"))
        
        for item in results.get("organic", []):
            link = item.get("link", "")
            ext = tldextract.extract(link)
            if ext.domain and ext.suffix:
                noise = ["youtube", "wikipedia", "linkedin", "facebook", "twitter", "amazon"]
                if ext.domain not in noise:
                    return f"{ext.domain}.{ext.suffix}"
        return None
    except Exception as e:
        print(f"   - Serper domain lookup failed for {company}: {e}")
        return None


# Well-known companies as ultimate fallback
WELL_KNOWN_DOMAINS = {
    "samsung": "samsung.com", "apple": "apple.com", "google": "google.com",
    "microsoft": "microsoft.com", "openai": "openai.com", "meta": "meta.com",
    "amazon": "amazon.com", "nvidia": "nvidia.com", "intel": "intel.com",
    "amd": "amd.com", "qualcomm": "qualcomm.com", "tesla": "tesla.com",
    "sony": "sony.com", "lg": "lg.com", "huawei": "huawei.com",
    "xiaomi": "xiaomi.com", "oneplus": "oneplus.com", "realme": "realme.com",
    "oppo": "oppo.com", "vivo": "vivo.com", "anthropic": "anthropic.com",
    "ibm": "ibm.com", "oracle": "oracle.com", "adobe": "adobe.com",
}


def _find_domain_serper_sync(company: str) -> str:
    """Sync version of Serper domain lookup for use with asyncio.to_thread."""
    import json
    import urllib.request
    serper_key = os.getenv("SERPER_API_KEY", "").strip()
    if not serper_key:
        return None
    try:
        data = json.dumps({"q": f"{company} official website", "num": 3}).encode("utf-8")
        req = urllib.request.Request(
            "https://google.serper.dev/search",
            data=data,
            headers={"X-API-KEY": serper_key, "Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            results = json.loads(resp.read().decode("utf-8"))
        for item in results.get("organic", []):
            link = item.get("link", "")
            ext = tldextract.extract(link)
            if ext.domain and ext.suffix:
                noise = ["youtube", "wikipedia", "linkedin", "facebook", "twitter", "amazon"]
                if ext.domain not in noise:
                    return f"{ext.domain}.{ext.suffix}"
        return None
    except Exception as e:
        print(f"   - Serper domain lookup failed for {company}: {e}")
        return None


async def find_domain_with_fallback(company: str, tavily_client) -> str:
    """Try Tavily → Google → Serper → well-known dict to find domain."""
    # 1. Try Tavily
    domain = await find_single_official_domain(company, tavily_client)
    if domain:
        print(f"   ✅ {company} → {domain} (via Tavily)")
        return domain

    # 2. Try Google CSE
    domain = await find_single_official_domain_google(company)
    if domain:
        print(f"   ✅ {company} → {domain} (via Google)")
        return domain

    # 3. Try Serper (sync function, call via to_thread)
    domain = await asyncio.to_thread(_find_domain_serper_sync, company)
    if domain:
        print(f"   ✅ {company} → {domain} (via Serper)")
        return domain

    # 4. Well-known fallback
    key = company.lower().strip()
    if key in WELL_KNOWN_DOMAINS:
        domain = WELL_KNOWN_DOMAINS[key]
        print(f"   ✅ {company} → {domain} (via well-known dict)")
        return domain

    print(f"   ⚠️ Could not find domain for {company}")
    return None


async def find_official_domains(companies: List[str]) -> List[str]:
    """Discovers official domains for a list of companies with multi-provider fallback."""
    if not companies:
        return []
    
    client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
    
    # Run all company lookups in parallel with fallback chain
    tasks = [find_domain_with_fallback(company, client) for company in companies]
    results = await asyncio.gather(*tasks)
    
    # Filter out None and deduplicate
    domains = list(dict.fromkeys([r for r in results if r]))
    return domains[:5]

async def url_discovery(state: ResearchState) -> ResearchState:
    """LangGraph node to discover trusted and company domains."""
    from utils.logger import section, info, success, warning, step, phase_progress
    
    section("URL Discovery", "🌐")
    query = state["original_query"]
    
    # 1. Extract companies
    step(1, 3, "Extracting companies from query")
    companies = await extract_companies(query)
    if companies:
        success(f"Identified: {', '.join(companies)}")
    else:
        warning("No companies identified from query")
    
    # 2. Discover official domains
    step(2, 3, "Discovering official domains (multi-provider)")
    company_domains = await find_official_domains(companies)
    if company_domains:
        success(f"Company domains: {', '.join(company_domains)}")
    else:
        warning("No company domains found")
    
    # 3. Category-based trusted domains
    step(3, 3, "Loading trusted domains by category")
    category = await detect_category(query)
    trusted_domains = get_domains_by_category(category)
    info(f"Category: {category} → {len(trusted_domains)} trusted domains")
    
    # Bug Fix: Remove official company domains from the trusted_domains list.
    # This prevents a domain like 'openai.com' from appearing in BOTH
    # company_domains (official lane) and trusted_domains (industry-news lane),
    # which would cause redundant searches and inflated result counts.
    official_set = set(company_domains)
    trusted_domains_filtered = [d for d in trusted_domains if d not in official_set]
    if len(trusted_domains_filtered) < len(trusted_domains):
        removed = [d for d in trusted_domains if d in official_set]
        print(f"   ✂️  Removed {len(removed)} domain(s) from trusted list (already in official): {removed}")

    state["company_domains"] = company_domains
    state["trusted_domains"] = trusted_domains_filtered[:5]
    # Store primary entity for entity-anchored trusted search queries.
    # extract_companies() already called extract_primary_entity() internally;
    # use the first company name as the primary entity. This is the company
    # that all trusted-source queries will be anchored to.
    state["primary_entity"] = companies[0] if companies else ""
    if state["primary_entity"]:
        info(f"Primary entity for trusted search: '{state['primary_entity']}'")

    return state

