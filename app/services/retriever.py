from app.services.vectorstore import get_vectorstore


def get_retriever(k: int = 5):
    """
    Standard retriever (used inside RAG chain).
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_kwargs={"k": k}
    )


def retrieve_with_scores(query: str, k: int = 5):
    """
    Retrieve documents with similarity scores for confidence calculation.
    Returns: List[(Document, score)]
    """
    vectorstore = get_vectorstore()
    return vectorstore.similarity_search_with_score(query, k=k)
