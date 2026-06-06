import requests
import json

from app.core.config import (
    OLLAMA_URL,
    MODEL_NAME,
    KEEP_ALIVE
)


class LLMRouter:

    def route(self, text: str):

        prompt = f"""
You are a STRICT classification system for a farming assistant.

You MUST return ONLY valid JSON.
No explanation. No markdown. No extra text.

Return format:
{{
  "intent": "greeting | plant_help | general_question",
  "response_style": "diagnosis | explanation | short_answer",
  "need_rag": true/false,
  "need_weather": true/false,
  "missing_info": ["plant", "symptoms", "pest", "nutrient_issue"]
}}

Rules:
- plant_help → diagnosis style
- general_question → explanation or short_answer
- greeting → short_answer
- Only set need_weather=true if plant_help AND weather is relevant
- Only set need_rag=true if factual farming knowledge is needed
- If missing key info for plant_help → include missing_info

User:
{text}
"""

        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": MODEL_NAME,
                    "prompt": prompt,
                    "stream": False,
                    "keep_alive": KEEP_ALIVE,
                    "options": {"temperature": 0.0}
                }
            )

            raw = response.json().get("response", "").strip()

            # safety cleanup
            raw = raw.split("```")[-1].strip()

            return json.loads(raw)

        except Exception:
            return {
                "intent": "general_question",
                "response_style": "short_answer",
                "need_rag": True,
                "need_weather": False,
                "missing_info": []
            }


llm_router = LLMRouter()