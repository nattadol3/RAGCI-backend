import requests
from langchain.embeddings.base import Embeddings
import os

# Get API key from environment variable
api_key = os.environ.get("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("Missing TOGETHER_API_KEY environment variable")

class HuggingFaceEmbeddings(Embeddings):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", api_key=api_key):
        self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def embed_query(self, text: str):
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": text})
        response.raise_for_status()  # Raise an error if the request fails
        return response.json()  # Return the embedding vector

def get_embedding_function():
    return HuggingFaceEmbeddings()
