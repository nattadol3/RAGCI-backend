from sentence_transformers import SentenceTransformer

class SentenceTransformerEmbeddings:
    def init(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def call(self, text: str) -> list:
        return self.model.encode(text).tolist()

def get_embedding_function():
    return SentenceTransformerEmbeddings("all-MiniLM-L6-v2")
