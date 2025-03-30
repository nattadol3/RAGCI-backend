from langchain_ollama import OllamaEmbeddings

def get_embedding_function():
    # embeddings = OpenAIEmbeddings()
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return embeddings