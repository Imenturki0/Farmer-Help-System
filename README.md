# 🌱 Farmer AI Assistant — Production RAG System for Agriculture

## Overview

Farmer AI Assistant is a Retrieval-Augmented Generation (RAG) system designed to provide reliable agricultural assistance using a private knowledge base.

The system combines modern AI techniques:

* Hybrid information retrieval
* Vector search
* Keyword search
* Neural reranking
* Local Large Language Models
* Incremental document ingestion

The goal is to build an AI assistant that can answer farming questions about:

* Crop diseases
* Fertilizers
* Soil management
* Nutrient deficiencies
* Irrigation
* Agricultural practices

---

# 🏗️ System Architecture

```
                 User
                   |
                   v

              FastAPI API

                   |
                   v

            Orchestrator Layer

        +----------+----------+
        |                     |
        v                     v

     Weather              RAG Pipeline


                         |
                         v

              Hybrid Retrieval

          +--------------+--------------+
          |                             |

       Qdrant                        BM25
   Vector Search              Keyword Search


          \                             /

                   RRF Fusion

                       |
                       v

              FlashRank Reranker

                       |
                       v

               Context Selection

                       |
                       v

              Llama3 (Ollama)

                       |
                       v

                 Final Answer
```

---

# 🚀 Main Features

## 🔎 Advanced RAG Pipeline

Unlike simple chatbot systems, Farmer AI uses:

### Dense Retrieval

Using:

* BAAI BGE embeddings
* Qdrant vector database

### Sparse Retrieval

Using:

* BM25 keyword search

### Hybrid Ranking

Combining both approaches using:

* Reciprocal Rank Fusion (RRF)

### Neural Reranking

Using:

* FlashRank reranker

to improve context quality before generation.

---

# 📚 Knowledge Ingestion Pipeline

The system supports automatic document updates:

```
New PDF
   |
File Watcher
   |
Change Detection
   |
PDF Processing
   |
Embedding Generation
   |
Qdrant Update
```

Features:

* New document detection
* Modified document detection
* Deleted document removal
* Incremental updates

---

# 🧠 Local LLM Deployment

The system uses:

* Ollama
* Llama3

Advantages:

* Private inference
* No external API dependency
* Suitable for offline environments

---

# 📊 Evaluation Framework

The project includes retrieval and generation evaluation.

Implemented metrics:

### Retrieval

* Recall@K
* Mean Reciprocal Rank (MRR)
* Retrieval latency

### Generation

* ROUGE-L
* Exact Match

Evaluation compares:

* Dense retrieval
* Hybrid retrieval
* Reranking approaches

---

# 🛠️ Technology Stack

## Backend

* Python
* FastAPI

## AI

* Sentence Transformers
* BGE Embeddings
* FlashRank
* Llama3

## Databases

* Qdrant Vector Database
* BM25 Retriever

## Deployment

* Docker
* Docker Compose

---

# 📂 Project Structure

```
farmer-helper

├── app
│   ├── core
│   │   ├── orchestrator.py
│   │   ├── router.py
│   │   └── memory.py
│   │
│   ├── services
│   │   ├── rag.py
│   │   ├── vector_db.py
│   │   ├── llm.py
│   │   └── weather.py
│   │
│   └── eval
│       ├── evaluate_rag.py
│       └── evaluate_retrieval.py
│
├── ingestion
│   ├── pipeline.py
│   └── watcher.py
│
├── data
│
├── docker-compose.yml
└── README.md
```

---

# ⚙️ Running the Project

## Start infrastructure

```bash
docker compose up -d
```

## Start Ollama

Install Ollama and run:

```bash
ollama pull llama3
```

## Start API

```bash
uvicorn app.main:app --reload
```

API:

```
http://localhost:8000
```

---

# 🎯 Engineering Highlights

This project demonstrates:

✅ Production-style RAG architecture
✅ Hybrid retrieval optimization
✅ Vector database integration
✅ Local LLM deployment
✅ Incremental data pipelines
✅ Retrieval evaluation
✅ Latency monitoring
✅ Containerized deployment

---

# Future Improvements

* RAGAS evaluation
* Automated CI/CD pipeline
* Authentication layer
* Monitoring dashboard
* Mobile deployment for offline farming assistance

---

# Author

Imen Turki

Machine Learning Engineer | Generative AI | RAG Systems




```
farmer-helper
├─ .pytest_cache
│  ├─ CACHEDIR.TAG
│  ├─ README.md
│  └─ v
│     └─ cache
│        ├─ lastfailed
│        └─ nodeids
├─ app
│  ├─ config
│  │  ├─ prompts.yaml
│  │  └─ settings.py
│  ├─ core
│  │  ├─ citations.py
│  │  ├─ config.py
│  │  ├─ logger.py
│  │  ├─ memory.py
│  │  ├─ orchestrator.py
│  │  ├─ orchestrator_old.py
│  │  └─ router.py
│  ├─ eval
│  │  ├─ build_eval_dataset.py
│  │  ├─ evaluate_rag.py
│  │  ├─ evaluate_rag_old.py
│  │  ├─ generation_eval.py
│  │  ├─ ragas_eval.py
│  │  ├─ retrieval_eval.py
│  │  └─ utils.py
│  ├─ main.py
│  ├─ main_old.py
│  ├─ middleware
│  │  └─ middleware_production.py
│  ├─ schemas.py
│  └─ services
│     ├─ bm25_retriever.py
│     ├─ llm.py
│     ├─ rag.py
│     ├─ rag_old.py
│     ├─ vector_db.py
│     └─ weather.py
├─ data
│  ├─ eval
│  │  ├─ generation_results
│  │  │  ├─ by_question_type.json
│  │  │  ├─ by_topic.json
│  │  │  ├─ production_details.json
│  │  │  └─ production_summary.json
│  │  ├─ groups.json
│  │  ├─ qa_dataset.json
│  │  └─ results
│  │     ├─ bm25_queries.json
│  │     ├─ bm25_summary.json
│  │     ├─ hybrid_queries.json
│  │     ├─ hybrid_summary.json
│  │     ├─ production_queries.json
│  │     ├─ production_summary.json
│  │     ├─ qdrant_queries.json
│  │     └─ qdrant_summary.json
│  ├─ processed
│  │  ├─ chunks.json
│  │  ├─ farming_docs.txt
│  │  └─ file_state.json
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
│  ├─ file_state.py
│  ├─ pipeline.py
│  ├─ prepare_data.py
│  └─ watcher.py
├─ logs
│  └─ app.json
├─ README.md
├─ requirements.txt
├─ scripts
│  └─ check_quality_gates.py
├─ tests
│  └─ test_orchestrator.py
└─ vector_db

```