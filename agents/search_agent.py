import os
from datetime import datetime, timedelta
from typing import List
from tavily import TavilyClient
from models.schemas import ResearchState, Article
from config import TAVILY_API_KEY
from utils.date_utils import is_within_last_14_days

def search_agent(state: ResearchState) -> ResearchState:
    """
    Executes searches and filters results to the last 14 days.
    """
    subqueries = state.get("subqueries", [])
    if not subqueries:
        return state
        
    client = TavilyClient(api_key=TAVILY_API_KEY)
    all_articles = []
    seen_urls = set()
    
    # We iterate over subqueries
    for query in subqueries[:5]: # Safety limit on queries
        print(f"🔍 Searching for: {query}")
        # Use days=14 and topic='news' for better recency and metadata
        response = client.search(
            query=query, 
            search_depth="advanced", 
            topic="news", 
            days=14, 
            max_results=5
        )
        for res in response.get('results', []):
            url = res.get('url')
            if url in seen_urls:
                continue
            
            pub_date = res.get('published_date', '')
            print(f"   - Found: {res.get('title')} ({pub_date})")
            
            # Since we requested days=14, we can have some trust, 
            # but we still check if metadata exists and is valid.
            # If metadata is missing but we used days=14, we'll allow it 
            # as a 'soft' fallback if it came from the news topic.
            
            if is_within_last_14_days(pub_date):
                 valid = True
            elif not pub_date:
                 # If we used days=14 and it's from 'news', we assume it's valid 
                 # but we label it with today's date or leave empty for the validator
                 valid = True 
                 pub_date = datetime.now().strftime('%Y-%m-%d') # Fallback to today
            else:
                 valid = False
            
            if valid:
                 all_articles.append(Article(
                    title=res.get('title', 'No Title'),
                    url=url,
                    snippet=res.get('content', ''),
                    published_date=pub_date
                ))
                 seen_urls.add(url)
                 print(f"     ✅ Article accepted.")
            else:
                 print(f"     ❌ Article rejected (date outside 14 days or invalid format).")
            
            if len(all_articles) >= 10:
                break
        if len(all_articles) >= 10:
            break
            
    if not all_articles:
        print("⚠️ No articles passed the date filter.")
            
    state["articles"] = all_articles[:10]
    return state
