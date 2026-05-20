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
    mode: str = "general"


@app.post("/ask")
def ask_farm(question: Question):
    user_text = question.text
    mode = getattr(question, "mode", "general")

    prompt = f"""
You are a farming expert.

Mode: {mode}

If mode is:
- crop → focus on crop health
- pest → focus on insects/disease
- soil → soil advice
- water → irrigation advice
- general → general farming help

User question:
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

    return {"answer": data.get("response", "No response")}
