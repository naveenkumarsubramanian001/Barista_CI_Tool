import asyncio
import tldextract
from typing import List
from tavily import TavilyClient
from models.schemas import (
    ResearchState,
    CompanyList,
    SuggestedCompanies,
    OfficialDomainSelection,
)
from config import TAVILY_API_KEY, get_llm
from utils.predefinedurls import detect_category, get_domains_by_category
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Initialize LLM
llm = get_llm()

# --- Batch Validation Chain ---
validation_parser = JsonOutputParser(pydantic_object=CompanyList)
validation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity classification system. Filter the provided list and return ONLY the items that are actual companies, organizations, or brands.",
        ),
        ("user", "Entities: {entities}\n\n{format_instructions}"),
    ]
)
validation_chain = validation_prompt | llm | validation_parser

# --- Dynamic Suggestion Chain ---
suggestion_parser = JsonOutputParser(pydantic_object=SuggestedCompanies)
suggestion_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an entity extraction expert. Based on the user's query, identify ONLY the primary companies, brands, or organizations explicitly mentioned or directly targeted by the query.\nCRITICAL RULES:\n1. If the user mentions a specific company (e.g., 'OpenAI', 'Anthropic'), ONLY return those.\n2. DO NOT list competitors or unrelated market leaders unless explicitly requested.\n3. Use the search context only to resolve partial names to their official company names.\n4. Return an empty list if no specific company is targeted.\nEnsure the generated output is a valid JSON with a 'companies' array containing string values.",
        ),
        (
            "user",
            "Search Context: {context}\n\nQuery: {query}\n\n{format_instructions}",
        ),
    ]
)
suggestion_chain = suggestion_prompt | llm | suggestion_parser

# --- Official Domain Selection Chain ---
selection_parser = JsonOutputParser(pydantic_object=OfficialDomainSelection)
selection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a brand verification expert. From the provided list of search results, identify the ONE URL that is the primary, official homepage of the company. Official sites include localized versions (e.g., /in, /global, /en) and regional subdomains. Avoid support pages, news articles, or social media if a main homepage is available. If no result looks like an official company website, return is_official=False and official_url=None.",
        ),
        (
            "user",
            "Company: {company}\nCandidates:\n{candidates}\n\n{format_instructions}",
        ),
    ]
)
selection_chain = selection_prompt | llm | selection_parser


async def validate_companies_batch(entities: List[str]) -> List[str]:
    """Validates a list of entities as companies in a single LLM call."""
    if not entities:
        return []
    try:
        result = await validation_chain.ainvoke(
            {
                "entities": ", ".join(entities),
                "format_instructions": validation_parser.get_format_instructions(),
            }
        )
        return result.get("companies", [])
    except Exception as e:
        print(f"   - Error in batch validation for {entities}: {e}")
        return []


async def suggest_companies_dynamic(query: str) -> List[str]:
    """Suggests relevant companies for a generic query using real-time search and LLM."""
    try:
        # 1. Search for real-world context using Tavily (running sync code in thread)
        client = TavilyClient(api_key=TAVILY_API_KEY)
        search_results = await asyncio.to_thread(
            client.search,
            query=f"official company website or companies related to: {query}",
            max_results=5,
        )

        results_list = search_results.get("results", [])
        print(f"   - Tavily context search found {len(results_list)} results.")

        context = "\n".join([f"Result: {r.get('content', '')}" for r in results_list])
        print(f"   --- DEBUG: Context Snippet ---\n{context[:1000]}...")

        # 2. Extract companies from context using LLM
        result = await suggestion_chain.ainvoke(
            {
                "query": query,
                "context": context,
                "format_instructions": suggestion_parser.get_format_instructions(),
            }
        )
        print(f"   --- DEBUG: LLM Raw result: {result}")
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


async def extract_companies(query: str, entities: List[str] = None) -> List[str]:
    """
    Extracts and validates companies.
    1. If specific entities are provided, validates them.
    2. If no companies found, dynamically suggests them based on query intent.
    """
    companies = []

    # 1. Batch validate entities if present
    if entities:
        companies = await validate_companies_batch(entities)

    # 2. Dynamic Suggestion if no companies identified yet
    if not companies:
        print("   - Using LLM to dynamically suggest relevant companies...")
        companies = await suggest_companies_dynamic(query)

    # Deduplicate and limit
    return list(dict.fromkeys(companies))[:5]


async def find_single_official_domain(company: str, client: TavilyClient) -> str:
    """Helper to find domain for a single company."""
    try:
        # 1. Search for top 3 candidates
        search_query = f"{company} official website"
        response = await asyncio.to_thread(
            client.search, query=search_query, max_results=8
        )
        results = response.get("results", [])

        if not results:
            return None

        # 2. Let LLM pick the best official one
        candidate_text = "\n".join(
            [f"- {r.get('title')}: {r.get('url')}" for r in results]
        )
        print(f"   --- DEBUG: Domain candidates for {company} ---\n{candidate_text}")
        selection = await selection_chain.ainvoke(
            {
                "company": company,
                "candidates": candidate_text,
                "format_instructions": selection_parser.get_format_instructions(),
            }
        )
        print(f"   --- DEBUG: Domain selection result for {company}: {selection}")

        official_url = selection.get("official_url")
        if not selection.get("is_official") or not official_url:
            return None

        # 3. Robust domain extraction using tldextract
        ext = tldextract.extract(official_url)
        if ext.domain and ext.suffix:
            root_domain = f"{ext.domain}.{ext.suffix}"

            # Final noise filter
            noise_subdomains = ["support", "news", "blog", "developer"]
            noise_domains = [
                "youtube",
                "wikipedia",
                "linkedin",
                "facebook",
                "twitter",
                "instagram",
                "tiktok",
                "amazon",
                "ebay",
            ]

            if (
                not any(sub in ext.subdomain for sub in noise_subdomains)
                and ext.domain not in noise_domains
            ):
                return root_domain
            else:
                print(
                    f"   - Filtered out {root_domain} for {company} due to noise match."
                )
    except Exception as e:
        print(f"   - Error finding domain for {company}: {str(e)[:200]}")
    return None


async def find_official_domains(companies: List[str]) -> List[str]:
    """Discovers official domains for a list of companies in parallel."""
    if not companies:
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)

    # Run all company searches in parallel
    tasks = [find_single_official_domain(company, client) for company in companies]
    results = await asyncio.gather(*tasks)

    # Filter out None and deduplicate
    domains = list(dict.fromkeys([r for r in results if r]))
    return domains[:5]


async def url_discovery(state: ResearchState) -> ResearchState:
    """LangGraph node to discover trusted and company domains."""
    state.setdefault("logs", [])
    state["logs"].append("🌐 Starting advanced URL discovery...")

    # 1. Extract companies (Dynamic + Batch Validation)
    state["logs"].append("🏢 Identifying target companies and brands...")
    companies = await extract_companies(state["original_query"])
    if companies:
        state["logs"].append(f"📍 Identified: {', '.join(companies)}")

    # 2. Discover official domains in parallel
    state["logs"].append("🔗 Searching for official company websites...")
    company_domains = await find_official_domains(companies)
    if company_domains:
        state["logs"].append(f"🏠 Mapped {len(company_domains)} official domains.")

    # 3. Category-based trusted domains
    category = await detect_category(state["original_query"])
    state["logs"].append(f"📊 Market category: {category}")
    trusted_domains = get_domains_by_category(category)

    # Update state
    state["company_domains"] = company_domains
    state["trusted_domains"] = trusted_domains[:5]

    return state
