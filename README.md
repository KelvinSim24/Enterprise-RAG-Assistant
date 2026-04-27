# 🏢 Nexus | Enterprise-Grade Local RAG Assistant

A robust, private, and fault-tolerant Retrieval-Augmented Generation (RAG) application. This project moves beyond "demo-ware" by implementing asynchronous orchestration, deterministic data versioning, and a completely local inference stack.

---
## HomePage of RAG Application
<img width="2549" height="1242" alt="image" src="https://github.com/user-attachments/assets/4de3fceb-f66a-466f-abfe-c3f2546522d5" />


## Questions and Answered tested with accurate result
<img width="1894" height="777" alt="image" src="https://github.com/user-attachments/assets/5d3527aa-0e28-4ac5-a234-517f959b5aa9" />
<img width="1923" height="967" alt="image" src="https://github.com/user-attachments/assets/2241c8a6-c89d-49f4-8ddd-4aa28ad40761" />

## Inngest Server (Orchestrator) - Saved and Audited with every event happened within the application usage.
<img width="2549" height="1242" alt="image" src="https://github.com/user-attachments/assets/86c44231-7147-4674-9de0-f5dfb0f829a0" />

## 🧠 System Architecture & Techniques

This application demonstrates several advanced software engineering patterns:

* **Asynchronous Orchestration (Inngest):** Unlike standard RAG apps that fail on LLM timeouts, this uses an event-driven architecture. Ingestion and Querying are handled as managed "steps," allowing for automatic retries and state recovery.
* **Vector Database Engineering (Qdrant):** Implements deterministic UUID generation (via `uuid5`) for vector IDs, ensuring "Upsert" idempotency and preventing duplicate data entries.
* **Local-First Privacy:** Utilizes **Ollama** for both embeddings (`qwen3-embedding:4b`) and text generation (`gemma4:e4b`), ensuring zero data leakage of sensitive documents.
* **Type-Safe Contracts:** Built with **Pydantic** to enforce strict data schemas across the entire pipeline, from Streamlit to the vector store.
* **Contextual Grounding:** Low-temperature settings and strict system prompting to eliminate hallucinations and ensure factual precision.

---

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **Orchestration:** Inngest (Local Dev Server)
- **Vector DB:** Qdrant (Dockerized)
- **Embedding Model:** `qwen3-embedding:4b`
- **LLM Engine:** Ollama (`gemma4:e4b`)
- **Frontend:** Streamlit with Custom CSS injection

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Docker Desktop installed.
- Ollama installed.
- Python 3.10+

### 2. Start Infrastructure
## Terminal 1: Vector Database
docker-compose up -d

## Terminal 2: Ollama Models
ollama pull gemma4:e4b

ollama pull qwen3-embedding:4b
