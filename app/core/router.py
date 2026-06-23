import json 
from app.services.llm import generate_answer

def routing_prompt (text: str):
    return f"""

You are a routing system for a farming assistant.

Classify the user question into ONE category:

- rag: farming knowledge needed
- weather: weather/climate request
- chat: greetings or casual conversation
- unknown: not agriculture related

Return ONLY JSON:

{{
 "route": "rag|weather|chat|unknown",
 "confidence": 0.0
}}

Question:
{text}
"""
def llm_route(text: str):
    
    response = generate_answer(
        routing_prompt(text)
    )

    try:
        data = json.loads(response)

        return (
            data.get("route", "unknown"),
            float(data.get("confidence", 0))
        )

    except Exception:

        # fallback
        return "rag", 0.5