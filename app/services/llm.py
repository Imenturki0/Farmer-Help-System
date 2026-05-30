import requests
from app.core.config import (
    OLLAMA_URL,
    MODEL_NAME,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    KEEP_ALIVE
)


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
                    "top_p": TOP_P
                }
        }
    )

    data = response.json()
    return data.get("response", "No response")