import json
import requests

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen3:4b",
        "prompt": "List 3 colors in JSON format.",
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_ollama()
