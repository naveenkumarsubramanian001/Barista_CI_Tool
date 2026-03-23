import json
import asyncio
import sys
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Import shared LLM factory from config
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from config import get_llm
from models.schemas import DecomposedQueries,SubQuery

# Import QueryAnalyzer from Step 1 (entity.py)
sys.path.append(os.path.dirname(__file__))
from utils.entity import QueryAnalyzer
from models.schemas import ResearchState

# ---------------------------------------------------------------------------
# LangGraph Node Interface
# ---------------------------------------------------------------------------

def decomposer_agent(state: ResearchState) -> ResearchState:
    """
    LangGraph Node for Query Decomposition.
    Includes relevance alignment validation to ensure subqueries
    are semantically relevant to the original query.
    """
    query = state.get("original_query", "")
    if not query:
        state["error"] = "No original query found in state."
        return state

    # Step 1: Analyze (Entities/Intent)
    analyzer = QueryAnalyzer()
    analysis = analyzer.analyze(query)

    decomposer = QueryDecomposer()
    
    # Inject feedback from state into analysis for the decomposer
    analysis["feedback"] = state.get("validation_feedback", "")
    
    result = decomposer.decompose(analysis)

    # Update State
    subqueries = [sq.get("subquery") if isinstance(sq, dict) else sq for sq in result.get("subqueries", [])]
    
    # =====================================================
    # Relevance Alignment Validation
    # Ensures search agents get queries relevant to the original,
    # not random tangents.
    # =====================================================
    try:
        from config import get_embedding_model
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        embed_model = get_embedding_model()
        query_emb = np.array(embed_model.embed_documents([query]))
        sub_embs = np.array(embed_model.embed_documents(subqueries))
        
        sims = cosine_similarity(query_emb, sub_embs)[0]
        
        RELEVANCE_THRESHOLD = 0.4
        aligned = []
        for i, sq in enumerate(subqueries):
            if sims[i] >= RELEVANCE_THRESHOLD:
                aligned.append(sq)
            else:
                print(f"   ⚠️ Removed divergent subquery (sim={sims[i]:.3f}): {sq}")
        
        if len(aligned) >= 3:
            subqueries = aligned
        else:
            print(f"   ⚠️ Only {len(aligned)} aligned subqueries — keeping all to avoid data loss")
            
    except Exception as e:
        print(f"   ⚠️ Relevance alignment check failed: {e} — using all subqueries")
    
    state["subqueries"] = subqueries
    
    return state

# ---------------------------------------------------------------------------
# Pydantic Output Schema
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

class QueryDecomposer:
    """
    Step 2: Query Decomposition Agent.

    Takes the structured JSON output from Step 1 (QueryAnalyzer / entity.py)
    and decomposes the original query into 3-5 focused, non-overlapping
    subqueries ready for parallel web search.
    """

    INTENT_STRATEGY = {
        "recommendation": "Break into: criteria subquery, top-options subquery, region/constraint subquery, review/rating subquery",
        "comparison":     "Break into: one subquery per item being compared, plus a head-to-head/benchmark subquery",
        "explanation":    "Break into: definition subquery, use-case subquery, code/example subquery, limitations subquery",
        "search":         "Break into: broad overview subquery, specific detail subqueries per entity",
        "analysis":       "Break into: data/stats subquery, expert-opinion subquery, trend subquery",
        "purchase":       "Break into: pricing subquery, availability subquery, review subquery, alternative subquery",
    }

    def __init__(self):
        self.llm = get_llm()
        self.parser = JsonOutputParser(pydantic_object=DecomposedQueries)
        self.chain = self._build_chain()

    # ------------------------------------------------------------------
    # Chain Construction
    # ------------------------------------------------------------------

    def _build_chain(self):
        system_prompt = (
            "You are an expert query decomposition engine for an AI research assistant.\n\n"
            "Your job: decompose a user query into 3-5 focused, NON-OVERLAPPING subqueries "
            "that together comprehensively cover the original intent.\n\n"
            "Rules:\n"
            "- Each subquery must target a DIFFERENT aspect of the original query\n"
            "- Subqueries must be web-search-optimised (concise, specific, no filler words)\n"
            "- Use the provided strategy hint to guide the decomposition\n"
            "- Incorporate entities and constraints directly into subqueries where relevant\n"
            "- If constraints exist (budget, date, location) include them in the relevant subquery\n"
            "- If no entities are detected, derive key concepts from the key_terms list\n"
            "- Edge case — ambiguous query: generate broader coverage subqueries\n\n"
            "{format_instructions}"
        )

        user_prompt = (
            "Decompose the following analyzed query:\n\n"
            "Original Query  : {original_query}\n"
            "Intent          : {top_intent} (confidence: {intent_confidence})\n"
            "Strategy Hint   : {strategy}\n"
            "Detected Entities:\n{entities}\n"
            "Constraints     : {constraints}\n"
            "Key Terms       : {key_terms}\n\n"
            "PREVIOUS FEEDBACK (If any): {feedback}\n\n"
            "Generate 3-5 focused subqueries as valid JSON."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user",   user_prompt),
        ])

        return prompt | self.llm | self.parser

    # ------------------------------------------------------------------
    # Input Normalization Helpers
    # ------------------------------------------------------------------

    def _normalize_entities(self, entities: List) -> List[Dict]:
        """
        Accept both formats from entity.py:
          - list-of-tuples  : ["India", "GPE", 33, 38]
          - list-of-dicts   : {"text": "India", "label": "GPE"}
        """
        out = []
        for ent in entities:
            if isinstance(ent, (list, tuple)) and len(ent) >= 2:
                out.append({"text": ent[0], "label": ent[1]})
            elif isinstance(ent, dict):
                out.append({"text": ent.get("text", ""), "label": ent.get("label", "UNKNOWN")})
        return out

    def _extract_key_terms(self, pos_tags: List) -> List[str]:
        """
        Extract nouns, proper nouns, and adjectives from POS tags.
        Accepts both list-of-tuples (spaCy) and list-of-dicts.
        """
        key_pos = {"NOUN", "PROPN", "ADJ"}
        terms: List[str] = []
        for tag in pos_tags:
            if isinstance(tag, (list, tuple)) and len(tag) >= 2:
                if tag[1] in key_pos:
                    terms.append(tag[0])
            elif isinstance(tag, dict):
                if tag.get("pos") in key_pos:
                    terms.append(tag.get("text", ""))
        return list(dict.fromkeys(terms))  # deduplicate, preserve order

    def _build_invoke_payload(self, entity_output: Dict) -> Dict:
        """Assemble the chain invocation payload from a Step 1 output dict."""
        entities = self._normalize_entities(entity_output.get("entities", []))
        key_terms = self._extract_key_terms(entity_output.get("pos_tags", []))
        top_intent = entity_output.get("top_intent", "search")

        # Edge case: no entities and no key terms → use raw query as fallback entity
        if not entities and not key_terms:
            entities = [{"text": entity_output.get("original_query", ""), "label": "QUERY"}]

        return {
            "original_query":    entity_output.get("original_query", ""),
            "top_intent":        top_intent,
            "intent_confidence": entity_output.get("intent_confidence", 0.5),
            "strategy":          self.INTENT_STRATEGY.get(top_intent, "Break into distinct focused subqueries"),
            "entities":          json.dumps(entities, indent=2),
            "constraints":       json.dumps(entity_output.get("constraints") or {}, indent=2),
            "key_terms":         ", ".join(key_terms) if key_terms else "none detected",
            "feedback":          entity_output.get("feedback", "None. This is the first attempt."),
            "format_instructions": self.parser.get_format_instructions(),
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, entity_output: Dict) -> Dict:
        """Synchronously decompose a single entity.py output."""
        payload = self._build_invoke_payload(entity_output)
        result  = self.chain.invoke(payload)
        result["original_query"] = entity_output.get("original_query", "")
        return result

    async def _decompose_async(self, entity_output: Dict) -> Dict:
        """Async decomposition for a single query (used in parallel batch)."""
        payload = self._build_invoke_payload(entity_output)
        result  = await self.chain.ainvoke(payload)
        result["original_query"] = entity_output.get("original_query", "")
        return result

    async def decompose_parallel(self, entity_outputs: List[Dict]) -> List[Dict]:
        """
        Decompose multiple entity.py outputs in parallel using asyncio.gather.
        Failed individual tasks return an error dict instead of crashing the batch.
        """
        tasks = [self._decompose_async(output) for output in entity_outputs]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        results: List[Dict] = []
        for i, res in enumerate(raw_results):
            if isinstance(res, Exception):
                results.append({
                    "original_query": entity_outputs[i].get("original_query", ""),
                    "error": str(res),
                    "subqueries": [],
                    "strategy": "N/A",
                })
            else:
                results.append(res)
        return results


# ---------------------------------------------------------------------------
# Entry Point — Step 1 → Step 2 live pipeline
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # --- Raw queries (the only thing you need to provide) ---
    raw_queries = [
        "I am going to start a smartphone brand with phones under 10000 INR who are my competitors"
    ]

    print("=" * 80)
    print("STEP 1: Query Analysis (entity.py)")
    print("=" * 80)

    # Step 1: run QueryAnalyzer to get structured entity output
    analyzer = QueryAnalyzer()
    entity_outputs = []
    for q in raw_queries:
        result = analyzer.analyze(q)
        entity_outputs.append(result)
        print(f"\n  Query       : {q}")
        print(f"  Intent      : {result['top_intent']} (confidence: {result['intent_confidence']})")
        print(f"  Entities    : {[(e[0], e[1]) for e in result['entities']]}")
        print(f"  Constraints : {result['constraints']}")

    print("\n" + "=" * 80)
    print("STEP 2: Query Decomposition (parallel)")
    print("=" * 80)

    # Step 2: decompose all queries in parallel using Step 1 output
    decomposer = QueryDecomposer()
    results = asyncio.run(decomposer.decompose_parallel(entity_outputs))

    for result in results:
        print(f"\n  Original  : {result.get('original_query', '')}")
        print(f"  Strategy  : {result.get('strategy', '')}")
        subqueries = result.get("subqueries", [])
        if not subqueries:
            print(f"  ERROR     : {result.get('error', 'Unknown error')}")
            continue
        for idx, sq in enumerate(subqueries, 1):
            if isinstance(sq, dict):
                print(f"  [{idx}] {sq.get('subquery', '')}")
                print(f"       Purpose      : {sq.get('purpose', '')}")
                if sq.get("entity_focus"):
                    print(f"       Entity Focus : {sq['entity_focus']}")
        print("-" * 80)

