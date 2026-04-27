import requests
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)


def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if hasattr(d, "text") and d.text]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generates embeddings using the local Ollama API."""
    url = "http://localhost:11434/api/embed"
    payload = {
        "model": "qwen3-embedding:4b",
        "input": texts
    }

    # 120s timeout in case of very large batches
    response = requests.post(url, json=payload, timeout=120)
    response.raise_for_status()

    return response.json()["embeddings"]
