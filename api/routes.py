from fastapi import APIRouter
from pydantic import BaseModel
from rag.retriever import get_top_k_chunks
from openai import OpenAI
from models.prompt_template import DEFAULT_RAG_PROMPT
import os

client = OpenAI()
router = APIRouter()

class AskRequest(BaseModel):
    query: str

@router.get("/", tags=["Welcome"])
def index():
    return {"message": "Welcome to the RAG FAQ Assistant API!"}

@router.post("/ask", tags=["RAG"])
def ask_question(request: AskRequest):
    query = request.query
    top_docs = get_top_k_chunks(query)

    context = "\n\n".join([doc.page_content for doc in top_docs])
    prompt = DEFAULT_RAG_PROMPT(context, query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
    )

    answer = response.choices[0].message.content

    return {
        "query": query,
        "answer": answer,
        "context_used": [doc.page_content for doc in top_docs],
    }