import os
import requests
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_URL = os.getenv("OPENAI_BASE_URL") + "chat/completions"

def ask_gemini(prompt_text, model_name="gemini-2.5-flash"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {GEMINI_API_KEY}"
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_text}]
    }
    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
