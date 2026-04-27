from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import requests


def get_model_dimension(model_name="qwen3-embedding:4b") -> int:
    """Fires a tiny test prompt to Ollama to determine the exact output dimension."""
    url = "http://localhost:11434/api/embed"
    try:
        response = requests.post(
            url, json={"model": model_name, "input": ["test"]})
        response.raise_for_status()
        return len(response.json()["embeddings"][0])
    except:
        return 2560  # Safe fallback for qwen3-embedding:4b


class QdrantStorage:
    def __init__(self, url="http://localhost:6333", collection='docs'):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection

        if not self.client.collection_exists(self.collection):
            dim = get_model_dimension()
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim, distance=Distance.COSINE),
            )

    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i],
                              payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, top_k: int = 5):
        # Bulletproof Search: Tries modern query_points first, safely falls back to legacy search
        try:
            res = self.client.query_points(
                collection_name=self.collection,
                query=query_vector,
                limit=top_k,
                with_payload=True
            )
            results = res.points
        except AttributeError:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True
            )

        contexts = []
        sources = set()

        for r in results:
            payload = getattr(r, "payload", None)
            if payload is None and hasattr(r, "dict"):
                payload = r.dict().get("payload", {})
            if payload is None:
                payload = {}

            text = payload.get("text", "")
            source = payload.get("source", "")

            if text:
                contexts.append(text)
                sources.add(source)

        return {"contexts": contexts, "sources": list(sources)}
