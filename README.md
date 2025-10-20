RAG FAQ Assistant

A lightweight Retrieval-Augmented Generation (RAG) app built with OpenAI, FAISS, and FastAPI.

## Features

- Semantic FAQ Search using OpenAI Embeddings
- FastAPI backend with `/ask` endpoint
- FAISS Vector Search
- Env-based configuration
- Prompt templates
- Swagger docs: http://localhost:8000/docs

## Project Structure
rag-faq-assistant/
├── api/                  # FastAPI routes and entrypoint
├── rag/                  # Embedding & retrieval logic
├── models/               # Prompt templates & LLM wrappers
├── data/                 # FAQ files (e.g. txt, csv)
├── utils/                # Config, helper functions
├── main.py               # App entrypoint
├── requirements.txt      # Dependencies
├── .env.example          # API key placeholders
└── README.md             # Project overview

## Getting Started

```bash
git clone https://github.com/your-user/rag-faq-assistant.git
cd rag-faq-assistant
cp .env.example .env
# Edit .env and add your OpenAI API key
pip install -r requirements.txt
python rag/ingest.py
uvicorn main:app --reload

## Sample Query
POST /ask
{
  "query": "What is your return policy?"
}