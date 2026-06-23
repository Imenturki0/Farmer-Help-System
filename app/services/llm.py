import requests
from app.core.config import (
    OLLAMA_URL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    KEEP_ALIVE
)
import json


def generate_answer(prompt: str):

    response = requests.post(
         OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    
                }
        }
    )

    data = response.json()
    return data.get("response", "No response")

def generate_stream(prompt: str):
    
    response = requests.post(
         OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": True,
                "keep_alive": KEEP_ALIVE,
                "options": {
                    "temperature": TEMPERATURE,
                    "top_k": TOP_K,
                    "top_p": TOP_P,
                    
                }
        },
         stream=True
    )

    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode("utf-8"))
            yield data.get("response", "")