import pydantic
from typing import Optional, List  # <-- We must import List and Optional here!


class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: List[str]
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    ingested: int


class RAGSearchResult(pydantic.BaseModel):
    contexts: List[str]
    sources: List[str]


class RAGQueryResult(pydantic.BaseModel):
    answer: str
    sources: List[str]
    num_contexts: int
