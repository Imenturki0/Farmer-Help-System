from fastapi import FastAPI
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware
from rag import search_docs

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
    lat: float
    lon: float

# -----------------------------
# WEATHER
# -----------------------------
def get_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        res = requests.get(url, timeout=5)
        return res.json()
    except:
        return {"current_weather": {"temperature": 0, "windspeed": 0}}

# -----------------------------
# FORMAT CONTEXT (IMPORTANT FIX)
# -----------------------------
def format_context(context_list):
    return "\n".join(context_list[:3])

# -----------------------------
# MAIN ENDPOINT
# -----------------------------
@app.post("/ask")
def ask_farm(question: Question):

    user_text = question.text
    weather_data = get_weather(question.lat, question.lon)

    current = weather_data.get("current_weather", {})
    temp = current.get("temperature", 0)
    wind = current.get("windspeed", 0)

    context_list = search_docs(user_text)
    context = format_context(context_list)

    prompt = f"""
You are a very experienced farmer helping another farmer in real life.

IMPORTANT STYLE RULES:
- Speak naturally like a human (very important)
- 3 to 4 sentences only
- NO explanations like a textbook
- NO phrases like "most of the time", "usually", "it can be"
- Be direct and confident
- Choose ONE main cause only
- Give ONE clear action only
- Sound calm and practical, not formal

BEHAVIOR:
- If plant is dying → act urgent and simple
- Do not list multiple causes
- Do not hedge answers
- Do not sound like an AI assistant

Context:
{context}

Weather:
Temperature: {temp}°C
Wind: {wind} km/h

User question:
{user_text}
"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "gemma4:e2b",
            "prompt": prompt,
            "stream": False,
            "keep_alive": "5m",
            "options": {
                "temperature": 0.7,
                "top_k": 20,
                "top_p": 0.9
            }
        },
        timeout=60
    )

    data = response.json()

    return {"answer": data.get("response", "No response")}