import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3:latest")

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_K = int(os.getenv("TOP_K", "20"))
TOP_P = float(os.getenv("TOP_P", "0.9"))

KEEP_ALIVE = os.getenv("KEEP_ALIVE", "5m")