from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
import tempfile
import os

from app.services.memory import add_to_memory, get_memory
from app.services.rag_chain import get_rag_chain, rewrite_question, get_llm
from app.services.retriever import retrieve_with_scores, get_retriever

from app.utils.loader import load_pdf
from app.utils.splitter import split_docs

from app.services.vectorstore import get_vectorstore
from app.services.s3_service import upload_file_to_s3

router = APIRouter()

# Initialize once (important for performance)
rag_chain = get_rag_chain()
vectorstore = get_vectorstore()


# -------------------------------
# Request & Response Schemas
# -------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: str


class Source(BaseModel):
    document: str
    page: int | str
    confidence: float
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


# -------------------------------
# Chat Endpoint (RAG + Citations + Confidence)
# -------------------------------

from app.services.memory import add_to_memory, get_memory
from app.services.retriever import retrieve_with_scores
from app.services.rag_chain import rewrite_question, get_llm

@router.post("/chat", response_model=ChatResponse)
async def chat_with_documents(payload: ChatRequest):
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        session_id = payload.session_id

        # 1️⃣ Load memory
        history = get_memory(session_id)

        # 2️⃣ Rewrite question
        llm = get_llm()
        standalone_query = rewrite_question(llm, history, payload.query)

        # 3️⃣ Retrieve documents
        results = retrieve_with_scores(standalone_query, k=5)

        if not results:
            answer = "I could not find the answer in the provided documents."
            sources = []
        else:
            # 4️⃣ Generate answer
            answer = rag_chain.invoke(standalone_query)

            # 5️⃣ Build sources
            sources = []
            for doc, score in results:
                confidence = round(1 / (1 + score), 3)
                sources.append({
                    "document": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "N/A"),
                    "confidence": confidence,
                    "snippet": doc.page_content[:200]
                })

        # 6️⃣ Update memory
        add_to_memory(session_id, "user", payload.query)
        add_to_memory(session_id, "assistant", answer)

        return {
            "answer": answer,
            "sources": sources
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# @router.post("/chat", response_model=ChatResponse)
# async def chat_with_documents(payload: ChatRequest):
#     if not payload.query.strip():
#         raise HTTPException(status_code=400, detail="Query cannot be empty")

#     try:
#         # 1️⃣ Retrieve documents WITH similarity scores
#         results = retrieve_with_scores(payload.query, k=5)

#         if not results:
#             return {
#                 "answer": "I could not find the answer in the provided documents.",
#                 "sources": []
#             }

#         # 2️⃣ Generate answer using RAG chain
#         answer = rag_chain.invoke(payload.query)

#         # 3️⃣ Build citations with confidence
#         sources = []
#         for doc, score in results:
#             confidence = round(1 / (1 + score), 3)

#             sources.append({
#                 "document": doc.metadata.get("source", "unknown"),
#                 "page": doc.metadata.get("page", "N/A"),
#                 "confidence": confidence,
#                 "snippet": doc.page_content[:200]
#             })

#         return {
#             "answer": answer,
#             "sources": sources
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# # -------------------------------
# # PDF Upload + Indexing Endpoint
# # -------------------------------

@router.post("/upload-pdf")
async def upload_and_index_pdf(file: UploadFile = File(...)):
    """
    Upload PDF → S3 → Chunk → Embed → Pinecone
    """

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Upload to S3
        s3_url = upload_file_to_s3(tmp_path, file.filename)

        # Load & process PDF
        documents = load_pdf(tmp_path)
        chunks = split_docs(documents)

        # Store chunks in Pinecone
        vectorstore.add_documents(chunks)

        # Cleanup
        os.remove(tmp_path)

        return {
            "message": "PDF uploaded and indexed successfully",
            "s3_url": s3_url,
            "chunks_added": len(chunks),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
