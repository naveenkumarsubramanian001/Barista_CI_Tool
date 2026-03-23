"""Test Google Custom Search API key."""
import os
import json
import urllib.request
import urllib.parse
from dotenv import load_dotenv

load_dotenv()

key = os.getenv("GOOGLE_API_KEY", "")
cx = os.getenv("GOOGLE_CSE_ID", "")

print(f"API Key: {key[:10]}...{key[-4:]}" if len(key) > 14 else f"API Key: {key}")
print(f"CSE ID: {cx}")

params = urllib.parse.urlencode({
    "key": key, "cx": cx,
    "q": "Samsung One UI 8.5", "num": 3
})
url = f"https://www.googleapis.com/customsearch/v1?{params}"

try:
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        items = data.get("items", [])
        print(f"\nSUCCESS: Got {len(items)} results")
        for i, item in enumerate(items):
            print(f"  [{i+1}] {item.get('title')}")
            print(f"      {item.get('link')}")
except urllib.error.HTTPError as e:
    body = e.read().decode("utf-8") if e.readable() else ""
    print(f"\nHTTP Error {e.code}: {e.reason}")
    try:
        err_data = json.loads(body)
        error_detail = err_data.get("error", {})
        print(f"Message: {error_detail.get('message', 'N/A')}")
        print(f"Status: {error_detail.get('status', 'N/A')}")
        errors = error_detail.get("errors", [])
        for err in errors:
            print(f"  Domain: {err.get('domain')}")
            print(f"  Reason: {err.get('reason')}")
            print(f"  Message: {err.get('message')}")
    except:
        print(f"Raw body: {body[:500]}")
except Exception as e:
    print(f"\nError: {e}")
