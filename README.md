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
