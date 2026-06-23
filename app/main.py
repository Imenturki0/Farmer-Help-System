from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Question
from app.core.orchestrator import handle_question,handle_question_steam
from app.services.rag import rag
# from fastapi.responses import StreamingResponse
from fastapi.responses import PlainTextResponse
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# LOGGING (BASIC PROD LEVEL)
# -------------------------
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("farm-ai")


# -------------------------
# STARTUP EVENT (IMPORTANT)
# Instead of rebuilding embeddings every request.
# -------------------------

@app.on_event("startup")
def startup():
    logger.info("Loading RAG model...")

    rag.load_docs("data/processed/chunks.json")
    rag.build_index()

    logger.info("RAG ready ✔")


# -------------------------
# API
# -------------------------
@app.post("/ask")
def ask_farm(question: Question):
    try:
        answer = handle_question(question)
        return PlainTextResponse(
            content=answer
        )

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": "Internal server error"}
    
@app.post("/ask-stream")
def ask_stream_route(question: Question):
   return  handle_question_steam(question)