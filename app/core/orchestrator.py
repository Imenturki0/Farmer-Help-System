from app.services.weather import get_weather
from app.services.rag import rag
from app.services.llm import generate_answer,generate_stream
import json
from fastapi.responses import StreamingResponse
from app.core.router import llm_route
from app.core.memory import memory
# -------------------------
# FORMAT CONTEXT
# -------------------------
def format_context(results):
    if not results:
        return ""

    return "\n\n".join(r["text"] for r in results[:3])


    
def log_debug(data):
    with open("debug_logs.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

def finish(session_id, user_text, answer):
    memory.add(session_id, "User", user_text)
    memory.add(session_id, "Assistant", answer)
    return answer
# -------------------------
# MAIN PIPELINE
# -------------------------
def handle_question(question,session_id):


    text = question.text

    history = memory.get(session_id)

    history_text =""

    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"
    # -------------------------
    # 1. ROUTING
    # -------------------------
    route, confidence= llm_route(text)

    if confidence < 0.6:
        route = "rag"

    print(
    "ROUTE:",
    route,
    "CONF:",
    confidence
)


    rag_results = []
    best_score = -1
    context = ""
    weather = ""

    print("=" * 50)
    print("QUESTION:", text)
    print("ROUTE:", route)
    
    # -------------------------
    # 2. CHAT
    # -------------------------
    if route == "chat":
        answer= generate_answer(f"""
            You are a friendly farming assistant.
            
            Conversation History:
            {history_text}

            User said: {text}

            Respond naturally and briefly.
            """)
        return finish(session_id, text, answer)
     # -------------------------
    # 3. WEATHER
    # -------------------------
    if route == "weather":
        try:
            weather_data = get_weather(question.lat, question.lon)
            current = weather_data.get("current_weather", {})

            weather = f"""
            Temperature: {current.get('temperature', 0)}°C
            Wind: {current.get('windspeed', 0)} km/h
            """
        except Exception as e:
            print("Weather error:", e)

        answer= generate_answer(f"""
                You are a farming assistant.
                
                Conversation History:
                {history_text}

                Explain this weather for farming:
                {weather}
                """)
        return finish(session_id, text, answer)
    # -------------------------
    # 4. RAG
    # -------------------------
    if route == "rag":
        rag_results, best_score = rag.search(text, k=10, final_k=5)

        if rag_results and best_score > 0.3:
            context = "\n\n".join(r["text"] for r in rag_results[:3])
        else:
            context = ""
    # -------------------------
    # 5. UNKNOWN
    # -------------------------
    if route == "unknown":
        return generate_answer("""
       The user asked something outside farming.

Politely respond:
"I can only help with farming-related questions like crops, soil, fertilizers, irrigation, and plant diseases."
Do NOT use quotes.
Keep the response short and natural.
        """)
    # -------------------------
    # 6. FINAL PROMPT (RAG + fallback)
    # -------------------------
    prompt = f"""
    You are a friendly farming assistant.

    Conversation History:
    {history_text}

    RULES:
    - Only answer farming-related questions
    - Use the previous conversation when the user refers to something like "it", "that", "this", or "yes".
    - Do not hallucinate
    - Use context if available

    User Question:
    {text}

    Context:
    {context if context else "No relevant farming context found."}
    """
    log_debug({
        "question": text,
        "route": route,
        "best_score": best_score,
        "retrieved": [r["chunk_id"] for r in rag_results[:5]]
    })

    answer = generate_answer(prompt)

    return finish(session_id, text, answer)
   
  
def stream_with_memory(prompt , session_id ,user_text):

    full_answer =""

    for token in generate_stream(prompt):

        full_answer += token

        yield token 
    
    memory.add(session_id, "User", user_text)
    memory.add(session_id, "Assistant", full_answer)
    
   

def ask_stream(question):
    
    rag_results, best_score = rag.search(question.text, k=20, final_k=3)

    context = "\n\n".join(r["text"] for r in rag_results[:3])

    weather = ""

    prompt = f"""
You are a farming assistant.

User Question:
{question.text}

Context:
{context}
"""

    return StreamingResponse(
        generate_stream(prompt),
        media_type="text/plain"
    )


def handle_question_steam(question,session_id):

    text =question.text

    history = memory.get(session_id)

    history_text = ""

    for msg in history:
        history_text += f"{msg['role']}: {msg['content']}\n"

    route ,confidence= llm_route(text)

    if confidence < 0.6 :
        route ="rag"

    rag_results=[]
    context=""
    weather =""

    if route=="chat":
        prompt =f"""
you are friendly farming assistant.

Conversation History:
{history_text}

User said:{text}

respond naturally and briefly.
"""
        return StreamingResponse( 
            stream_with_memory(
                prompt,
                session_id,
                text
        ),
            media_type="text/plain")
    
    if route=="weather":
        weather_data=get_weather(question.lat,question.lon)
        current = weather_data.get("current_weather",{})

        weather = f"""
        Temperature: {current.get('temperature', 0)}°C
        Wind: {current.get('windspeed', 0)} km/h
        """

        prompt = f"""
        You are a farming assistant.

        Conversation History:
        {history_text}

        Explain this weather for farming:

        {weather}


        Current User Question:
        {text}
        """

        return StreamingResponse( 
            stream_with_memory(
                prompt,
                session_id,
                text
        ),
            media_type="text/plain")
    
    if route == "rag":
        rag_results, best_score = rag.search(text, k=10, final_k=5)

        if rag_results and best_score > 0.3:
            context = "\n\n".join(r["text"] for r in rag_results[:3])
        else:
            context = ""

    if route == "unknown":
        prompt = """
        You are a farming assistant.

        Politely say you only help with farming topics like crops, soil, fertilizers, irrigation, and plant diseases.

        Keep it short and natural.
        """
        return StreamingResponse( 
            stream_with_memory(
                prompt,
                session_id,
                text
        ),
            media_type="text/plain")

    prompt = f"""
    You are a friendly farming assistant.

    Conversation History:
    {history_text}

    RULES:
    - Only answer farming questions.
    - Use previous conversation when appropriate.
    - Do not hallucinate.

    User Question:
    {text}

    Context:
    {context if context else "No relevant farming context found."}
    """

    return StreamingResponse( 
            stream_with_memory(
                prompt,
                session_id,
                text
        ),
            media_type="text/plain")


# Question
#  |
# Router
#  |
# Intent decision
#  |
# Service execution
#  |
# Context building
#  |
# LLM generation
#  |
# Memory update

# Problem 1: Orchestrator is becoming too large ⚠️

# Currently:

# handle_question()

# does:

# routing
# weather
# RAG
# prompt building
# memory
# logging
# response formatting

# In production I would split:

# orchestrator.py

# router_handler.py
# rag_handler.py
# weather_handler.py
# chat_handler.py
# memory_manager.py