from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router


def create_app() -> FastAPI:
    """
    Application factory.
    Keeps app creation clean and testable.
    """

    app = FastAPI(
        title="rag-fastapi-gemini",
        description="RAG system using FastAPI, Gemini, Pinecone and LangChain",
        version="1.0.0"
    )

    # ----------------------------------
    # CORS (for future frontend usage)
    # ----------------------------------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # tighten this in prod
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ----------------------------------
    # API Routes
    # ----------------------------------
    app.include_router(
        router,
        prefix="/api"
    )

    # ----------------------------------
    # Health Check
    # ----------------------------------
    @app.get("/health")
    async def health_check():
        return {"status": "ok", "service": "rag-fastapi-gemini"}

    return app


# Create app instance
app = create_app()
