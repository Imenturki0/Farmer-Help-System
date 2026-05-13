from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
     allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    text: str

# @app.get("/")
# def home():
#     return {"message": "🌱 Farmer Helper API is running"}

@app.post("/ask")
def ask_farm(question: Question):
    user_text = question.text

    prompt = f"""
You are a helpful assistant.

You can talk about general topics and also help with farming.

If the user asks about farming (crops, soil, pests), give agricultural advice.
If not, respond normally like a friendly assistant.


User message:
{user_text}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma4:e2b",
            "prompt": prompt,
            "stream": False
        }
    )

    data = response.json()

    # 🔥 DEBUG (IMPORTANT)
    print("OLLAMA RAW:", data)

    # SAFE extraction
    answer = data["response"]

    if not answer:
        return {"answer": "AI did not return response"}

    return {"answer": answer}