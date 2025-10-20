from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_concise_summary_prompt():
    payload = {
        "query": "What is LangChain?",
        "prompt_style": "concise-summary"
    }
    response = client.post("/ask", json=payload)
    
    assert response.status_code == 200
    answer = response.json()["answer"]
    
    # Optional: Check that answer is relatively concise
    assert len(answer.split()) < 50  # tweak this based on your prompt behavior

def test_invalid_prompt_style():
    payload = {
        "query": "Tell me about vector databases.",
        "prompt_style": "foo-bar-style"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 400
    assert "Unknown prompt style" in response.json()["detail"]