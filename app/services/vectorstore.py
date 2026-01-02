from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from app.services.embeddings import get_embeddings
from app.config import settings

def get_vectorstore():
    pc = Pinecone(api_key=settings.PINECONE_API_KEY)
    index = pc.Index(settings.PINECONE_INDEX)

    return PineconeVectorStore(
        index=index,
        embedding=get_embeddings()
    )
