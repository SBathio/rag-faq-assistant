def DEFAULT_RAG_PROMPT(context: str, question: str) -> str:
    return f"""You are an intelligent assistant helping users with questions based on the following context.

Context:
--------
{context}

Question:
---------
{question}

Answer:
"""