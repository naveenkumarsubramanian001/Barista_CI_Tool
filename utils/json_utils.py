import re
import json


def safe_json_extract(text: str):
    """
    Extracts first valid JSON object from LLM output.
    Prevents failure if model adds extra commentary.
    """
    if not text:
        raise ValueError("Empty LLM response")

    # Standard cleanup for Qwen/Ollama common quirks
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try regex extraction
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in LLM response: {text[:100]}...")

    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"Found something like JSON but failed to parse: {str(e)}")
