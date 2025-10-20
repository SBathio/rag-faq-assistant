from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(
    title="RAG FAQ Assistant",
    description="A Retrieval-Augmented Generation API for answering FAQ queries using OpenAI and FAISS.",
    version="0.1.0"
)

# Register API routes
app.include_router(api_router)

@app.get("/health", tags=["Utility"])
def health_check():
    return {"status": "ok", "message": "RAG FAQ Assistant is running."}