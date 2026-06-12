from app.services.weather import get_weather
from app.services.rag import rag
from app.services.llm import generate_answer


# -------------------------
# FORMAT CONTEXT
# -------------------------
def format_context(results):
    if not results:
        return ""

    return "\n\n".join(r["text"] for r in results[:3])


# -------------------------
# RAG USAGE DECISION
# -------------------------
def should_use_rag(results):
    """
    Decide based on reranker confidence distribution,
    not a fake fixed threshold.
    """

    if not results:
        return False

    scores = [r.get("rerank_score", -999) for r in results]

    # best match quality
    best = max(scores)

    # if model is totally unsure
    if best < -3.0:
        return False

    return True


# -------------------------
# MAIN PIPELINE
# -------------------------
def handle_question(question):

    text = question.text

    # -------------------------
    # 1. RAG SEARCH
    # -------------------------
    rag_results, _ = rag.search(text, k=20, final_k=3)

    use_rag = should_use_rag(rag_results)

    print("\n" + "=" * 50)
    print("QUESTION:", text)
    if rag_results:
            print(f"BEST RERANK SCORE: {rag_results[0]['rerank_score']:.3f}")

    if use_rag:
        print("✅ RAG USED")
        context = format_context(rag_results)

        for i, r in enumerate(rag_results, 1):
            score = r.get("rerank_score", 0.0)
            print(f"\nChunk {i} | Score: {score:.3f}")
            print(r["text"])

    else:
        print("❌ RAG SKIPPED")
        context = ""

    print("=" * 50 + "\n")

    # -------------------------
    # 2. WEATHER (ONLY IF RAG USED)
    # -------------------------
    weather = ""

    if use_rag:
        try:
            weather_data = get_weather(question.lat, question.lon)
            current = weather_data.get("current_weather", {})

            weather = f"""
Temperature: {current.get('temperature', 0)}°C
Wind: {current.get('windspeed', 0)} km/h
"""
        except Exception as e:
            print("Weather error:", e)

    # -------------------------
    # 3. PROMPT (CLEANER + MORE CONTROLLED)
    # -------------------------
    prompt = f"""
You are a friendly farming assistant.

RULES:
- You are ONLY allowed to answer farming and agriculture-related questions
- If the question is not about farming, politely say:
  "I can only help with farming-related questions."
- Be friendly and helpful
- Do NOT hallucinate or guess
- Use context only if relevant

If the user greets you (hi, hello), respond

User Question:
{text}


Context:
{context if context else "No relevant farming context found."}

Weather:
{weather if weather else "Not relevant"}
"""

    # -------------------------
    # 4. GENERATE
    # -------------------------
    return generate_answer(prompt)