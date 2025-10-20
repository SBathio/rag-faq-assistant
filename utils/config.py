import os
from dotenv import load_dotenv
from openai import OpenAI

if not os.path.exists(".env"):
    raise FileNotFoundError(".env file not found. Please create one or copy from .env.example")

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
TOP_K = int(os.getenv("TOP_K", "3"))

VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "rag/vector_store.index")
DOC_STORE_PATH = os.getenv("DOC_STORE_PATH", "rag/docs.pkl")

OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_API_BASE,
    timeout=30
)

if not os.path.exists(VECTOR_STORE_PATH):
    raise FileNotFoundError(f"FAISS index not found at {VECTOR_STORE_PATH}. Please run rag/ingest.py first.")
if not os.path.exists(DOC_STORE_PATH):
    raise FileNotFoundError(f"Document store not found at {DOC_STORE_PATH}. Please run rag/ingest.py first.")