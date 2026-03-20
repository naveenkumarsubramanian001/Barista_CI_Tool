from models.schemas import CategorySelection
from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

DOMAIN_DB = {
    "technology": [
        "theverge.com",  # Strong reporting + insider sources
        "techcrunch.com",  # Startup + product leaks
        "wired.com",  # Deep investigative tech journalism
        "arstechnica.com",  # Highly trusted technical reporting
        "engadget.com",  # Early product news + leaks
    ],
    "ai": [
        "venturebeat.com",  # AI trends + insider coverage
        "syncedreview.com",  # AI research news (semi-insider)
        "analyticsindiamag.com",  # India-focused AI insights
        "towardsdatascience.com",  # Early discussions + trends (Medium-based)
    ],
    "leaks": [
        "gsmarena.com",  # Mobile leaks + specs (very reliable)
        "91mobiles.com",  # Frequent smartphone leaks (India strong)
        "androidauthority.com",  # Leaks + early hands-on
        "slashleaks.com",  # Raw leak aggregator
        "notebookcheck.net",  # Early hardware info + benchmarks
    ],
    "business_news": [
        "bloomberg.com",  # VERY strong for insider leaks
        "ft.com",  # Financial Times (corporate insights)
        "reuters.com",  # Verified but sometimes early signals
        "cnbc.com",  # Market + company insider info
        "businessinsider.com",  # Often breaks internal stories
    ],
    "enterprise_it": [
        "zdnet.com",  # Enterprise IT news
        "computerworld.com",  # Deep IT trends
        "crn.com",  # Channel and service providers (Crucial for Virtusa/TCS/etc)
        "cio.com",  # Tech leadership and enterprise IT
        "informationweek.com",  # Business technology
    ],
}

# Initialize LLM components
llm = get_llm()
category_parser = JsonOutputParser(pydantic_object=CategorySelection)
category_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a category classification expert. Given a user query and a list of available categories, identify the most relevant category. If the query is related to one of the categories (e.g., 'agents' is related to 'ai'), select that category. Available categories: {categories}",
        ),
        ("user", "Query: {query}\n\n{format_instructions}"),
    ]
)
category_chain = category_prompt | llm | category_parser


def get_domains_by_category(category: str):
    """Returns a list of trusted domains for a given category."""
    return DOMAIN_DB.get(category.lower(), [])


async def detect_category(query: str) -> str:
    """Detects the category of a query using LLM for semantic matching."""
    query_lower = query.lower()
    categories = list(DOMAIN_DB.keys())

    # 1. Simple keyword-based category detection (Fast path)
    if any(
        word in query_lower
        for word in ["smartphone", "laptop", "gadget", "mobile", "leak"]
    ):
        return "leaks"
    elif any(
        word in query_lower
        for word in ["ai", "machine learning", "deep learning", "llm"]
    ):
        return "ai"
    elif any(
        word in query_lower
        for word in [
            "services",
            "consulting",
            "enterprise",
            "it ",
            "bpo",
            "virtusa",
            "infosys",
            "tcs",
        ]
    ):
        return "enterprise_it"
    elif any(
        word in query_lower
        for word in ["business", "market", "stock", "finance", "revenue", "release"]
    ):
        return "business_news"

    # 2. Use LLM for semantic mapping (Fallback)
    try:
        result = await category_chain.ainvoke(
            {
                "query": query,
                "categories": ", ".join(categories),
                "format_instructions": category_parser.get_format_instructions(),
            }
        )
        category = result.get("category", "").lower()
        if category in categories:
            print(f"   - LLM semantically matched '{query}' to category: {category}")
            return category
    except Exception as e:
        print(f"   - Error in semantic category detection: {e}")

    return "technology"  # Default fallback
