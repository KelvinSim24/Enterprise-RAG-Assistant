""" 
Activation Command :
Terminal 1: docker run -p 6333:6333 -p 6334:6334 -v "${PWD}\qdrant_storage:/qdrant/storage" qdrant/qdrant
Terminal 2: uvicorn main:app --reload --port 8000
Terminal 3: npx inngest-cli@latest dev
Terminal 4: streamlit run streamlit_app.py
"""

import logging
import uuid
import os
import requests
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGChunkAndSrc, RAGQueryResult, RAGSearchResult, RAGUpsertResult

# Load environment variables
load_dotenv()

# PRODUCTION OPTIMIZATION: Validate crucial environment variables at startup
if not os.getenv("VOYAGE_API_KEY"):
    logging.warning(
        "VOYAGE_API_KEY is not set. Ingestion and Querying will fail.")

# Initialize Inngest Client (Ensured App ID matches Streamlit)
inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=os.getenv("ENVIRONMENT") == "production",
    serializer=inngest.PydanticSerializer()
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
    # PRODUCTION OPTIMIZATION: Add reasonable retries for ingestion
    retries=3
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        # Check if file actually exists before processing
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at path: {pdf_path}")

        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Call Voyage AI for embeddings
        vecs = embed_texts(chunks)

        # Generate deterministic UUIDs for Qdrant
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL,
                   f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]}
                    for i in range(len(chunks))]

        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    # Execute as orchestrated steps
    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)

    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    # Fixed typo to match frontend
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai"),
    retries=3
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    # 1. Extract Event Data
    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    # 2. Search Vector DB
    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k), output_type=RAGSearchResult)

    # 3. Construct Prompt
    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above. If the answer is not in the context, say 'I do not have enough information to answer that based on the provided documents.'"
    )

    # 4. Local Ollama LLM Call
    def _call_ollama(prompt: str) -> str:
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": "gemma4:e4b",  # Ensure this exactly matches your pulled Ollama model
            "messages": [
                {"role": "system", "content": "You are a helpful, precise assistant that answers questions strictly using the provided context."},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": 0.2,  # Low temperature for factual RAG
                "num_predict": 1024
            }
        }

        try:
            # PRODUCTION OPTIMIZATION: Always use timeouts for external API/HTTP calls
            response = requests.post(url, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()["message"]["content"].strip()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Failed to connect to local Ollama instance. Is it running?")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama inference failed: {str(e)}")

    # 5. Execute LLM Call as an Inngest Step
    answer = await ctx.step.run("llm-answer", lambda: _call_ollama(user_content))

    # Return formatted result
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    ).model_dump()


# Initialize FastAPI and serve Inngest functions
app = FastAPI(title="Production RAG API")

# Ensure BOTH functions are in the serve list
inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai])
