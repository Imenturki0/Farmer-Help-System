import requests
from app.config.settings import settings
import json
OLLAMA_URL = settings.llm.model_path
MODEL_NAME = settings.llm.model_name
TEMPERATURE = settings.llm.temperature
TOP_K = settings.llm.top_k
TOP_P = settings.llm.top_p
KEEP_ALIVE = settings.llm.keep_alive

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