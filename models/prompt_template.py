# models/prompt_template.py

from typing import Callable

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

def CONCISE_SUMMARY_PROMPT(context: str, question: str) -> str:
    return f"""Answer the user's question as concisely as possible using the provided context.

Context:
{context}

Q: {question}
A:"""

PROMPT_TEMPLATES: dict[str, Callable[[str, str], str]] = {
    "default": DEFAULT_RAG_PROMPT,
    "concise-summary": CONCISE_SUMMARY_PROMPT
}

def get_prompt_template(style: str) -> Callable[[str, str], str]:
    if style not in PROMPT_TEMPLATES:
        raise ValueError(
            f"Unknown prompt style: '{style}'. "
            f"Available styles: {list(PROMPT_TEMPLATES.keys())}"
        )
    return PROMPT_TEMPLATES[style]