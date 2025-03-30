from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def __call__(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def embed_query(self, query: str) -> list:
        """Encodes a single query string into an embedding."""
        return self.model.encode(query).tolist()

    def embed_documents(self, texts: list) -> list:
        """Encodes a list of documents into embeddings."""
        return self.model.encode(texts).tolist()

def get_embedding_function():
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
