import faiss
import pickle
import numpy as np
from rag.embedding import get_embedding
from utils.config import VECTOR_STORE_PATH, DOC_STORE_PATH
from utils.config import TOP_K

DEFAULT_RAG_PROMPT = """You are a helpful assistant. Answer the question using the provided context only.

Context:
{context}

Question: {question}

If the answer is not in the context, say you don't know."""

def get_top_k_chunks(index, query_embedding):
    return index.search(query_embedding, k=TOP_K)

def get_top_k_chunks(query: str, k: int = 3) -> list[str]:
    index = faiss.read_index(str(VECTOR_STORE_PATH))
    with open(DOC_STORE_PATH, "rb") as f:
        docs = pickle.load(f)

    query_embedding = get_embedding(query)
    _, indices = index.search(np.array([query_embedding]), k)
    return [docs[i] for i in indices[0] if i < len(docs)]