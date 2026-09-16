# 🌱 Farmer Helper

### An AI-powered agricultural assistant built around Hybrid RAG, Local LLMs, and Evaluation

Farmer Helper is a production-oriented AI assistant designed to answer agricultural questions using a curated knowledge base of agricultural documents.

The project goes beyond a basic RAG chatbot by focusing on the engineering problems that appear when building a more reliable AI system:

**hybrid retrieval, grounded generation, evaluation, checkpointing, streaming, observability, and failure analysis.**

---

## 🎯 Why I Built This

Agricultural information can be scattered across manuals, guides, and technical documents.

The goal of Farmer Helper is to turn this information into an assistant that can:

* retrieve relevant agricultural knowledge,
* answer questions using that evidence,
* avoid unsupported information when the knowledge base is insufficient,
* provide conversational interactions,
* and measure its own retrieval and generation quality.

The project started as a simple offline agricultural assistant and evolved into a **production-oriented RAG system**.

---

# 🧠 How It Works

```text
                         USER
                           │
                           ▼
                    ┌─────────────┐
                    │   FastAPI   │
                    │     API     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Router   │
                    └──────┬──────┘
                           │
             ┌─────────────┼─────────────┐
             │             │             │
             ▼             ▼             ▼
           CHAT         WEATHER          RAG
                                         │
                                         ▼
                              ┌────────────────────┐
                              │  Hybrid Retrieval  │
                              └─────────┬──────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         │                             │
                         ▼                             ▼
                    Dense Search                  BM25 Search
                      Qdrant                       Sparse
                         │                             │
                         └──────────────┬──────────────┘
                                        │
                                        ▼
                                  RRF Fusion
                                        │
                                        ▼
                               Top Relevant Chunks
                                        │
                                        ▼
                                  Local LLM
                                   Ollama
                                        │
                                        ▼
                      ┌─────────────────────────────┐
                      │   Citation Enforcement      │
                      │   (Adaptive Threshold)      │
                      └─────────────────────────────┘
                                        │
                                        ▼
                                  Grounded Answer
```

---

# 🔎 Retrieval Pipeline

Farmer Helper uses **hybrid retrieval** instead of relying on a single search method.

### Dense Retrieval

Agricultural documents are embedded using:

```text
BAAI/bge-base-en-v1.5
```

and stored in:

```text
Qdrant
```

Dense retrieval helps identify information that is semantically related to the user's question.

### Sparse Retrieval

The system also uses:

```text
BM25Okapi
```

BM25 is particularly useful for exact terminology such as:

* crop names,
* diseases,
* fertilizer names,
* measurements,
* technical terms.

### Hybrid Fusion

The results from both retrievers are combined using:

```text
Reciprocal Rank Fusion (RRF)
```

with:

```text
k = 60
```

This allows the system to combine semantic and lexical retrieval signals.

---

# 📚 Document Processing

Agricultural PDFs are processed using **Docling**.

```text
PDF
 │
 ▼
Docling
 │
 ▼
Structured extraction
 │
 ▼
Structure-aware chunking
 │
 ▼
Metadata
 │
 ▼
Embeddings
 │
 ▼
Qdrant
```

Current chunking configuration:

| Parameter          |      Value |
| ------------------ | ---------: |
| Maximum chunk size | ~250 words |
| Overlap            |  ~40 words |
| Minimum text       |  ~10 words |

Chunks contain metadata such as:

```text
chunk_id
source
title
pages
section
```

The ingestion pipeline also tracks file hashes so that changed and removed documents can be synchronized with the index.

---

# 🤖 Local LLM

Farmer Helper uses **Ollama** for local inference.

### Application model

```text
llama3:latest
```

Using a local model makes experimentation possible without depending on paid external LLM APIs.

It also gives the project control over:

* model selection,
* inference parameters,
* prompts,
* latency,
* and the evaluation environment.

---

# 🛡️ Grounded Generation

The RAG generation pipeline is explicitly instructed to answer using the retrieved context.

The model is told to:

* answer the exact question,
* use only retrieved information,
* avoid outside knowledge,
* avoid inventing numbers or recommendations,
* avoid answering a different question,
* acknowledge insufficient evidence,
* mention conflicting information when necessary.

If the retrieved documents do not contain enough information, the assistant can respond:

> "I don't have enough information in the provided sources to answer this."

This makes **grounding** an explicit part of the generation pipeline rather than assuming that retrieval automatically guarantees a correct answer.

---

## 🔐 Citation Enforcement (Adaptive)

**NEW:** Farmer Helper now uses adaptive citation thresholds instead of a fixed threshold.

### The Problem

A fixed citation threshold (0.7) was rejecting valid answers that were well-supported by the context. This caused:

```text
Correct context retrieved ✓
Good answer generated ✓
Citation check → threshold=0.7 ✗
Result: "I don't have enough information"
```

### The Solution

Adaptive thresholds based on retrieval confidence:

```python
if best_score > 0.8:
    threshold = 0.5   # High confidence retrieval → lenient
elif best_score > 0.5:
    threshold = 0.65  # Medium confidence → moderate
else:
    threshold = 0.75  # Low confidence → strict
```

### Impact

* Fewer false rejections of valid answers
* Better balance between accuracy and coverage
* More natural conversations
* Improved evaluation metrics

---

# ⚡ Streaming

The API supports streaming responses.

Instead of waiting for the complete answer:

```text
Request
   ↓
Retrieve
   ↓
Generate
   ↓
Stream chunks
   ↓
Complete response
```

This is particularly useful when running an LLM locally, where generation can take noticeable time.

---

# 💬 Conversation & Routing

Farmer Helper is not limited to a single RAG endpoint.

Requests can be routed to:

```text
chat
weather
rag
unknown
```

The system also maintains bounded conversation history per session.

This allows follow-up questions while preventing conversation history from growing indefinitely.

---

# 🌦️ Weather

The assistant can retrieve weather information and provide it to the LLM as context.

Current weather data includes:

```text
Temperature
Wind speed
```

This creates a second information source outside the agricultural document collection.

---

# 📊 Evaluation

One of the main goals of this project is to **measure the system instead of assuming it works**.

Evaluation is performed at multiple levels.

---

## 🔎 Retrieval Evaluation

Current baseline:

| Metric            |   Result |
| ----------------- | -------: |
| Recall@1          |   0.2117 |
| Recall@3          |   0.3150 |
| Recall@5          |   0.3767 |
| Recall@10         |   0.4567 |
| Hit Rate@5        |   0.7700 |
| MRR               |   0.6541 |
| NDCG@5            |   0.4004 |
| Retrieval latency | ~4.56 ms |

These numbers come from the current evaluation dataset and are treated as a **project baseline**, not as a general benchmark.

The important question is not simply:

> "Is the score good?"

but:

> "What type of failure produced this score?"

---

## 🧪 LLM Evaluation

Generated answers are evaluated using **DeepEval**.

Current metrics:

### Faithfulness

Does the answer remain supported by the retrieved context?

### Answer Relevancy

Does the answer actually address the user's question?

The evaluation uses a separate local judge model:

```text
qwen2.5:3b
```

running through Ollama.

### Evaluation Improvements

**NEW:** Enhanced evaluation pipeline with error recovery:

* **Timeout handling** - 60-second timeout per question with graceful degradation
* **Ollama health checks** - Detects when Ollama crashes and auto-restarts
* **Cascading failure prevention** - Single bad answer no longer breaks subsequent evaluations
* **Quality validation** - Skips empty/error answers before evaluation
* **Checkpointing** - Progress saved after each evaluation (atomic writes)

This ensures that long-running evaluations (100+ questions) complete reliably.

---

## 🔬 Evaluation Architecture

```text
                  Evaluation Dataset
                    (Synthetic QA)
                         │
                         ▼
         ┌───────────────────────────────┐
         │  1. Generate RAG Results      │
         │  (with checkpoints)           │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  2. Validate Generated Answers│
         │  (skip empty/error)           │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  3. Evaluate with DeepEval    │
         │  (with timeout/recovery)      │
         └───────────────┬───────────────┘
                         │
         ┌───────────────┴────────────┐
         │                            │
         ▼                            ▼
    Faithfulness            Answer Relevancy
         │                            │
         └───────────────┬────────────┘
                         ▼
                    Evaluation
                      Results
```

An important design decision is that **generation and evaluation are separated**.

The application model generates the answer.

A separate model evaluates the answer.

---

# 💾 Checkpointed Evaluation

LLM evaluation can be slow, especially when everything runs locally.

Instead of keeping results only in memory, the evaluation pipeline checkpoints its progress.

```text
generated_rag_results.json
faithfulness.json
answer_relevancy.json
deepeval_summary.json
```

This means an interrupted evaluation does not necessarily require starting from zero.

The pipeline also uses:

* **Atomic file writes** - Reduces risk of corrupting checkpoints
* **Checkpoint recovery** - Resumes from last checkpoint on restart
* **Per-question checkpointing** - Saves progress after each evaluation

---

# 🧩 Failure Analysis

A major engineering lesson from this project is that **not every bad answer has the same cause**.

### Retrieval failure

```text
Question
   ↓
Incorrect / missing context
   ↓
Incorrect answer
```

### Generation failure

```text
Question
   ↓
Correct context
   ↓
LLM
   ↓
Incorrect answer
```

### Citation enforcement failure

```text
Question
   ↓
Correct context retrieved
   ↓
Correct answer generated
   ↓
Citation threshold too strict
   ↓
Answer rejected (false negative)
```

### Evaluation failure

```text
Good answer
   ↓
LLM Judge
   ↓
Incorrect evaluation
```

This distinction is important.

For example, improving the embedding model will not solve a problem where the correct context was already retrieved but the LLM ignored it.

---

# 🏥 Reliability

The FastAPI application includes:

* Request IDs
* Request logging
* Error logging
* Exception handling
* Health checks
* Deep dependency checks
* Startup/shutdown logging
* Graceful degradation under load

Endpoints include:

```text
GET  /health
GET  /health/deep
GET  /info
GET  /metrics
POST /ask
POST /ask-stream
```

---

# 🛠️ Tech Stack

| Area                | Technology             |
| ------------------- | ---------------------- |
| Language            | Python                 |
| API                 | FastAPI                |
| LLM Runtime         | Ollama                 |
| Application LLM     | Llama 3                |
| Evaluation LLM      | Qwen 2.5 3B            |
| Embeddings          | BAAI/bge-base-en-v1.5  |
| Vector Database     | Qdrant                 |
| Sparse Retrieval    | BM25Okapi              |
| Fusion              | Reciprocal Rank Fusion |
| Document Processing | Docling                |
| Evaluation          | DeepEval               |
| API Documentation   | OpenAPI / Swagger      |
| Version Control     | Git / GitHub           |

---

# 📁 Project Structure

```text
farmer-helper/
│
├── app/
│   ├── config/
│   │   ├── prompts.yaml
│   │   └── settings.py
│   │
│   ├── core/
│   │   ├── citations.py          ← Adaptive threshold
│   │   ├── logger.py
│   │   ├── memory.py
│   │   ├── orchestrator.py       ← Adaptive threshold logic
│   │   └── router.py
│   │
│   ├── eval/
│   │   ├── config.py
│   │   ├── build_eval_dataset.py ← Synthetic data generation
│   │   ├── deepeval_eval.py      ← Timeout/recovery logic
│   │   ├── retrieval_eval.py
│   │   ├── runner.py
│   │   └── dataset.py
│   │
│   ├── services/
│   │   ├── bm25_retriever.py
│   │   ├── llm.py
│   │   ├── rag.py
│   │   ├── vector_db.py
│   │   └── weather.py
│   │
│   └── main.py
│
├── ingestion/
│   ├── prepare_data.py
│   ├── file_state.py
│   ├── pipeline.py
│   └── watcher.py
│
├── data/
│   ├── raw/
│   │   └── pdfs/
│   └── processed/
│
├── tests/
│
├── .deepeval/
├── requirements.txt
└── README.md
```

---

# 🚀 Getting Started

## Requirements

* Python 3.10+
* Ollama
* Qdrant

## Clone

```bash
git clone https://github.com/Imenturki0/Farmer-Help-System.git

cd Farmer-Help-System
```

## Create environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

## Install dependencies

```bash
pip install -r requirements.txt
```

## Pull the application model

```bash
ollama pull llama3:latest
```

## Pull the evaluation model

```bash
ollama pull qwen2.5:3b
```

## Start Qdrant

Run Qdrant locally on:

```text
http://localhost:6333
```

## Add documents

Place agricultural PDFs inside:

```text
data/raw/pdfs/
```

Then run the ingestion pipeline.

## Start the API

```bash
uvicorn app.main:app --reload
```

FastAPI will expose the interactive API documentation through its OpenAPI/Swagger interface.

## Run Evaluation

### Build synthetic evaluation dataset

```bash
python -m app.eval.build_eval_dataset
```

This generates Q&A pairs from your document chunks using LLM-based synthetic data generation with quality validation.

### Run full evaluation pipeline

```bash
python -m app.eval.runner
```

This will:
1. Generate answers using the RAG system
2. Evaluate retrieval quality
3. Evaluate generation quality (faithfulness, relevancy)
4. Checkpoint progress after each step
5. Produce evaluation reports

Evaluation typically takes 15-25 minutes for 100 questions depending on LLM speed.

---

# ⚠️ Current Limitations

Farmer Helper is still under active development.

Current limitations include:

* Local LLM inference can be slow.
* Qdrant is currently running locally.
* BM25 is maintained in memory.
* Conversation memory is process-local.
* Authentication and authorization are not implemented.
* Production deployment infrastructure is not yet implemented.
* LLM-as-a-judge evaluation has inherent limitations.
* Retrieval quality still requires improvement.
* Observability is currently basic.
* The system has not yet been validated at large scale with real-world users.

These limitations are intentionally documented rather than hidden.

---

# 🛣️ Roadmap

### Evaluation

* [x] Retrieval evaluation
* [x] Faithfulness evaluation
* [x] Answer relevancy evaluation
* [x] Evaluation checkpointing
* [x] Adaptive citation thresholds
* [x] Evaluation error recovery
* [ ] Detailed failure analysis
* [ ] Improve retrieval based on failure patterns
* [ ] Re-run evaluation after improvements

### Reliability

* [ ] Automated regression tests
* [ ] Better error handling
* [ ] Stronger citation verification
* [ ] Retrieval confidence calibration
* [ ] More robust session management

### Production

* [ ] Dockerize services
* [ ] Persistent application state
* [ ] Production Qdrant deployment
* [ ] Observability and tracing
* [ ] CI/CD
* [ ] Load testing
* [ ] Security hardening
* [ ] Production deployment

---

# 🧠 Engineering Takeaways

Building Farmer Helper taught me that making a RAG system **work** is only the beginning.

Some of the practical problems encountered during development were:

* Retrieval quality cannot be assumed from a working vector database.
* Hybrid retrieval needs measurable evaluation.
* RRF scores are ranking signals, not probability values.
* Retrieving the correct context does not guarantee a correct answer.
* Fixed citation thresholds can reject valid answers (false negatives).
* Adaptive thresholds based on retrieval confidence improve balance.
* LLM judges can also produce unreliable evaluations.
* Local LLM evaluation can become a significant latency bottleneck.
* Long-running evaluation requires checkpointing and recovery.
* Evaluation should follow the same generation path as the production application.
* Streaming changes how latency and generation are handled.
* Observability is essential for understanding failures.

The project is therefore being developed as an **evaluated and production-oriented AI system**, rather than simply a chatbot demo.

---

# 👩‍💻 About

**Imen Turki**

Machine Learning Engineer | AI / LLM / GenAI

Focus:

`RAG` · `LLMs` · `NLP` · `Hybrid Retrieval` · `AI Evaluation` · `Python` · `FastAPI`

### Links

**GitHub:**
https://github.com/Imenturki0/Farmer-Help-System

**Portfolio:**
https://imenturki0.github.io/My-Portfolio
