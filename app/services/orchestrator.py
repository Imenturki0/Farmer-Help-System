from app.services.weather import get_weather
from app.services.rag import rag
from app.services.llm import generate_answer


def format_context(context_list):
    if not context_list:
        return ""

    return "\n".join(context_list[:3])


def handle_question(question):

    # 1. WEATHER
    weather_data = get_weather(question.lat, question.lon)

    current = weather_data.get("current_weather", {})
    temp = current.get("temperature", 0)
    wind = current.get("windspeed", 0)

    # 2. RAG
    context_list = rag.search(question.text)
    context = format_context(context_list)

    # 3. FRIENDLY FARMING PROMPT (IMPORTANT CHANGE)
#     prompt = f"""
# You are a friendly farming assistant and experienced farmer.

# Your goal:
# - Give practical, simple farming advice like a helpful friend
# - Be honest and realistic
# - Focus on actions the farmer can take today

# Rules:
# - Use provided context if it helps
# - If context is weak or missing, still give general farming advice
# - Do NOT refuse to answer
# - Do NOT sound robotic
# - Keep answer 3–6 sentences
# - Be practical and direct

# Context:
# {context if context else "No specific farming records available."}

# Weather:
# Temperature: {temp}°C
# Wind: {wind} km/h

# User question:
# {question.text}

# Answer:
# """
    prompt = f"""
You are a friendly farming expert assistant.

Your job:
Help farmers diagnose plant problems and give safe, practical advice.

IMPORTANT RULES:
- Think step by step before answering
- Do NOT guess aggressively
- If unsure, use "most likely"
- Avoid contradictory instructions
- Be safe and practical

Respond in this format:

Problem:
Cause:
Advice:
Warning (if needed):

Context:
{context if context else "No specific farming data available."}

Weather:
Temperature: {temp}°C
Wind: {wind} km/h

User question:
{question.text}

Answer:
"""

    # 4. GENERATE (slightly more stable)
    return generate_answer(prompt)