# 🏢 Enterprise RAG Assistant

A production-grade, fault-tolerant Retrieval-Augmented Generation (RAG) application.

Unlike standard RAG demos, this project is built for enterprise reliability. It utilizes **Inngest** for workflow orchestration (handling API rate limits, timeouts, and automatic retries), **Voyage AI** for state-of-the-art vector embeddings, a local **Qdrant** database for retrieval, and a local **Ollama (Gemma)** model to ensure complete data privacy during text generation.

## 🧠 System Architecture

1. **Frontend:** A premium, session-aware chat UI built with Streamlit.
2. **Orchestration:** Inngest handles the asynchronous workflows. If a third-party API (like Voyage AI) drops the connection, Inngest automatically queues and retries the exact failing step without crashing the app.
3. **Ingestion Pipeline:** Pydantic validates the data schema -> LlamaIndex parses and chunks PDFs -> Voyage AI generates 2048-dimensional embeddings -> Qdrant stores the vectors.
4. **Query Pipeline:** User queries are embedded via Voyage AI -> Searched against Qdrant via Cosine Similarity -> Pushed to a local Ollama instance for secure, private text generation.

## 🛠️ Tech Stack

- **Backend:** Python, FastAPI
- **Orchestration:** Inngest
- **Vector Database:** Qdrant (Dockerized)
- **Embedding Model:** Voyage AI (`voyage-4-large`)
- **LLM Engine:** Ollama (`gemma:9b`)
- **Frontend:** Streamlit (Custom CSS injected)

---

## 🚀 Quick Start Guide

### 1. Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- [Ollama](https://ollama.com/) installed with the Gemma 9B model pulled (`ollama pull gemma:9b`).
- Node.js installed (required for the Inngest Dev Server).
- Python 3.10+

### 2. Environment Setup

Clone the repository and create a `.env` file in the root directory:

```env
VOYAGE_API_KEY=your_voyage_api_key_here
ENVIRONMENT=development
```
