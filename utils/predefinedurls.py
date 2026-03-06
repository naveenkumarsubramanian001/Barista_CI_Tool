import asyncio
from typing import List
from models.schemas import CategorySelection
from config import get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

DOMAIN_DB = {
    "technology": [
        "gsmarena.com",
        "androidauthority.com",
        "theverge.com",
        "techcrunch.com",
        "wired.com"
    ],
    "ai": [
        "arxiv.org",
        "paperswithcode.com",
        "huggingface.co",
        "openai.com",
        "deepmind.com"
    ],
    "education": [
        "coursera.org",
        "edx.org",
        "khanacademy.org",
        "skitstraa.com"
    ],
    "news": [
        "reuters.com",
        "bbc.com",
        "nytimes.com",
        "theguardian.com"
    ]
}

# Initialize LLM components
llm = get_llm()
category_parser = JsonOutputParser(pydantic_object=CategorySelection)
category_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a category classification expert. Given a user query and a list of available categories, identify the most relevant category. If the query is related to one of the categories (e.g., 'agents' is related to 'ai'), select that category. Available categories: {categories}"),
    ("user", "Query: {query}\n\n{format_instructions}")
])
category_chain = category_prompt | llm | category_parser

def get_domains_by_category(category: str):
    """Returns a list of trusted domains for a given category."""
    return DOMAIN_DB.get(category.lower(), [])

async def detect_category(query: str) -> str:
    """Detects the category of a query using LLM for semantic matching."""
    query_lower = query.lower()
    categories = list(DOMAIN_DB.keys())
    
    # 1. Simple keyword-based category detection (Fast path)
    if any(word in query_lower for word in ["smartphone", "laptop", "tech", "gadget", "mobile"]):
        return "technology"
    elif any(word in query_lower for word in ["ai", "machine learning", "deep learning", "llm"]):
        return "ai"
    elif any(word in query_lower for word in ["course", "education", "learn", "school", "university"]):
        return "education"
    elif any(word in query_lower for word in ["news", "latest", "breaking", "update"]):
        return "news"

    # 2. Use LLM for semantic mapping (Fallback)
    try:
        result = await category_chain.ainvoke({
            "query": query,
            "categories": ", ".join(categories),
            "format_instructions": category_parser.get_format_instructions()
        })
        category = result.get("category", "").lower()
        if category in categories:
            print(f"   - LLM semantically matched '{query}' to category: {category}")
            return category
    except Exception as e:
        print(f"   - Error in semantic category detection: {e}")
        
    return "technology" # Default fallback
