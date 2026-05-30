from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import Question
from app.services.orchestrator import handle_question
from app.services.rag import rag

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
# -------------------------
@app.on_event("startup")
def startup():
    logger.info("Loading RAG model...")

    rag.load_docs("data/processed/farming_docs.txt")
    rag.build_index()

    logger.info("RAG ready ✔")


# -------------------------
# API
# -------------------------
@app.post("/ask")
def ask_farm(question: Question):
    try:
        answer = handle_question(question)
        return {"answer": answer}

    except Exception as e:
        logger.error(f"Error: {e}")
        return {"error": "Internal server error"}