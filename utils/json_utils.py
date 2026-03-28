"""
Robust JSON extraction from LLM output.
Handles markdown fences, thinking tags, trailing commas,
and other common LLM output quirks.
"""

import re
import json


def safe_json_extract(text: str) -> dict:
    """
    Extracts the first valid JSON object from LLM output.
    
    Handles:
    - Direct JSON (clean output)
    - Markdown code fences (```json ... ```)
    - <think>...</think> reasoning tags (Qwen, DeepSeek)
    - Extra commentary before/after JSON
    - Trailing commas in JSON
    - Multiple JSON blocks (takes the largest valid one)
    """
    if not text:
        raise ValueError("Empty LLM response")

    text = text.strip()

    # Step 1: Remove <think>...</think> blocks (reasoning models)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Step 2: Remove markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()

    # Step 3: Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Step 4: Try fixing trailing commas and parse again
    cleaned = _fix_trailing_commas(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Step 5: Extract all JSON-like blocks and try each (largest first)
    candidates = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if not candidates:
        # Try a greedy match for deeply nested JSON
        greedy = re.search(r"\{.*\}", text, re.DOTALL)
        if greedy:
            candidates = [greedy.group()]

    # Sort by length descending — the largest block is most likely the full JSON
    candidates.sort(key=len, reverse=True)

    for candidate in candidates:
        for attempt in [candidate, _fix_trailing_commas(candidate)]:
            try:
                parsed = json.loads(attempt)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue

    raise ValueError(
        f"No valid JSON object found in LLM response. "
        f"First 200 chars: {text[:200]}..."
    )


def _fix_trailing_commas(text: str) -> str:
    """Remove trailing commas before closing braces/brackets (common LLM quirk)."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text
