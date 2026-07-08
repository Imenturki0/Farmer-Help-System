# 📚 RAG System (FAISS + BM25 + Reranker + LLM)

## 🚀 Overview

This project is a Retrieval-Augmented Generation (RAG) system designed to answer agricultural questions using PDF-based knowledge.

It combines:
- 🔍 FAISS (dense vector search)
- 🔤 BM25 (keyword-based retrieval)
- ⚖️ Cross-Encoder reranker
- 🧠 LLM (Llama3 via Ollama) for final answer generation

---

## 🧱 System Architecture

User Question  
→ Query Embedding (SentenceTransformer)  
→ Retrieval (FAISS + BM25)  
→ Hybrid Merge  
→ Cross-Encoder Reranking  
→ Top-K Context Selection  
→ LLM Generation (Ollama / Llama3)  
→ Final Answer  

---

## 📊 Features

- Hybrid retrieval (FAISS + BM25)
- Reranking using CrossEncoder
- Evaluation framework
  - Recall@K
  - MRR
  - Exact Match
  - ROUGE-L
- Latency benchmarking
- PDF chunk-based knowledge base
- Local LLM inference (Ollama)

---

## 📈 Evaluation Metrics
Recall@K → measures retrieval success
MRR → ranking quality
Exact Match → answer correctness
ROUGE-L → similarity with ground truth

---

## 🧪 Evaluation Modes
FAISS only
BM25 + FAISS
FAISS + Reranker
FAISS + BM25 + Reranker

```
farmer-helper
├─ app
│  ├─ core
│  │  ├─ config.py
│  │  ├─ orchestrator.py
│  │  ├─ router.py
│  │  └─ state.py
│  ├─ eval
│  │  ├─ build_eval_dataset.py
│  │  ├─ clustering.py
│  │  ├─ evaluate_rag.py
│  │  └─ evaluate_retrieval.py
│  ├─ main.py
│  ├─ schemas.py
│  └─ services
│     ├─ bm25_retriever.py
│     ├─ llm.py
│     ├─ prompts.py
│     ├─ rag.py
│     └─ weather.py
├─ data
│  ├─ eval
│  │  ├─ groups.json
│  │  └─ qa_dataset.json
│  ├─ processed
│  │  ├─ chunks.json
│  │  └─ farming_docs.txt
│  └─ raw
│     ├─ crops.csv
│     ├─ fertilizers.csv
│     ├─ nutrients.csv
│     ├─ pdfs
│     │  ├─ 23. Dr Toe Toe Khaing (243-250).pdf
│     │  ├─ A-Guide-to-Vegetable-Growing---9th-Edition.pdf
│     │  ├─ Agronomy-Manual.pdf
│     │  ├─ Application_of_Plant_Fertilizer_Serum_Using_Natura.pdf
│     │  ├─ compost_factsheet_public.pdf
│     │  ├─ Fertilizers___Food_Waste_-_Jane_Goodall-s_Roots___Shoots.pdf
│     │  ├─ IJSRA-2025-0410Article.pdf
│     │  └─ Liquid-Fertilizer-Recipes-PDF.pdf
│     ├─ pesticides.csv
│     └─ temperature.csv
├─ docker-compose.yml
├─ dockerfile
├─ frontend
│  └─ index.html
├─ git
├─ ingestion
│  └─ prepare_data.py
├─ README.md
├─ requirements.txt
└─ vector_db

```
```
farmer-helper
├─ app
│  ├─ core
│  │  ├─ config.py
│  │  ├─ orchestrator.py
│  │  ├─ router.py
│  │  └─ state.py
│  ├─ eval
│  │  ├─ build_eval_dataset.py
│  │  ├─ clustering.py
│  │  ├─ evaluate_rag.py
│  │  └─ evaluate_retrieval.py
│  ├─ main.py
│  ├─ schemas.py
│  └─ services
│     ├─ bm25_retriever.py
│     ├─ llm.py
│     ├─ prompts.py
│     ├─ rag.py
│     ├─ vector_db.py
│     └─ weather.py
├─ data
│  ├─ eval
│  │  ├─ groups.json
│  │  └─ qa_dataset.json
│  ├─ processed
│  │  ├─ chunks.json
│  │  └─ farming_docs.txt
│  └─ raw
│     ├─ crops.csv
│     ├─ fertilizers.csv
│     ├─ nutrients.csv
│     ├─ pdfs
│     │  ├─ 23. Dr Toe Toe Khaing (243-250).pdf
│     │  ├─ A-Guide-to-Vegetable-Growing---9th-Edition.pdf
│     │  ├─ Agronomy-Manual.pdf
│     │  ├─ Application_of_Plant_Fertilizer_Serum_Using_Natura.pdf
│     │  ├─ compost_factsheet_public.pdf
│     │  ├─ Fertilizers___Food_Waste_-_Jane_Goodall-s_Roots___Shoots.pdf
│     │  ├─ IJSRA-2025-0410Article.pdf
│     │  └─ Liquid-Fertilizer-Recipes-PDF.pdf
│     ├─ pesticides.csv
│     └─ temperature.csv
├─ docker-compose.yml
├─ dockerfile
├─ frontend
│  └─ index.html
├─ git
├─ ingestion
│  └─ prepare_data.py
├─ README.md
├─ requirements.txt
└─ vector_db

```