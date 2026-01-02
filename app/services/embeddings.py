from langchain_huggingface import HuggingFaceEmbeddings

def get_embeddings():
    """
    2025-recommended embedding model for RAG.
    - Free
    - Fast
    - Works perfectly with Pinecone
    """

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
