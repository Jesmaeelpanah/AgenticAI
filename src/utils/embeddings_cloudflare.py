import os
import requests
from dotenv import load_dotenv

load_dotenv()

CLOUDFLARE_EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL")
CLOUDFLARE_API_TOKEN = os.getenv("EMBEDDING_API_KEY")
CLOUDFLARE_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")
CLOUDFLARE_EMBEDDING_URL = f"{CLOUDFLARE_EMBEDDING_BASE_URL}/embeddings"

def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {CLOUDFLARE_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": CLOUDFLARE_MODEL_NAME,
        "input": [text]
    }
    response = requests.post(CLOUDFLARE_EMBEDDING_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()['data'][0]['embedding']
