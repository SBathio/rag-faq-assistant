import os
import numpy as np
import faiss
import openai
import pickle
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from tiktoken import get_encoding
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

# Load env vars
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_DIR = Path("data")
INDEX_PATH = Path("rag/vector_store.index")
DOC_STORE_PATH = Path("rag/docs.pkl")

# Text splitter config
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " "],
)

def load_documents_from_txt(data_dir: Path) -> List[Document]:
    docs = []
    for file_path in data_dir.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")
        chunks = splitter.create_documents([text])
        docs.extend(chunks)
    return docs

def embed_documents(docs: List[Document]):
    print(f"Embedding {len(docs)} chunks...")
    embeddings = OpenAIEmbeddings()
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])
    return vectors

def build_faiss_index(vectors: List[List[float]], docs: List[Document]):
    dimension = len(vectors[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(vectors).astype("float32"))

    # Save index and documents
    faiss.write_index(index, str(INDEX_PATH))
    with open(DOC_STORE_PATH, "wb") as f:
        pickle.dump(docs, f)

    print(f" Saved index to: {INDEX_PATH}")
    print(f" Saved docs to: {DOC_STORE_PATH}")

if __name__ == "__main__":
    documents = load_documents_from_txt(DATA_DIR)
    embeddings = embed_documents(documents)
    build_faiss_index(embeddings, documents)