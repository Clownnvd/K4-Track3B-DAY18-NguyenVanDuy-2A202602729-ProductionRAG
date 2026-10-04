"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)  # noqa: E402


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    from underthesea import word_tokenize
    return word_tokenize(text, format='text').replace('_', ' ')


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        from rank_bm25 import BM25Okapi
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(c['text']).casefold().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens) if any(self.corpus_tokens) else None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0 or not query.strip():
            return []
        tokens = segment_vietnamese(query).casefold().split()
        scores = self.bm25.get_scores(tokens)
        order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))[:top_k]
        return [SearchResult(self.documents[i]['text'], float(scores[i]),
                             dict(self.documents[i].get('metadata', {})), 'bm25')
                for i in order if scores[i] > 0]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        try:
            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=2)
            self.client.get_collections()
        except Exception:
            self.client = QdrantClient(":memory:")
        self._encoder = None
        self._query_vectors = {}
        self.backend = 'memory' if getattr(self.client, '_init_options', {}).get('location') == ':memory:' else 'qdrant'

    def _get_encoder(self):
        if self._encoder is None:
            from src.model_cache import sentence_model
            self._encoder = sentence_model(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        from qdrant_client.models import Distance, VectorParams, PointStruct
        if self.client.collection_exists(collection):
            self.client.delete_collection(collection)
        self.client.create_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        if not chunks:
            return
        vectors = self._get_encoder().encode([c['text'] for c in chunks],
                                             normalize_embeddings=True, batch_size=4,
                                             show_progress_bar=True)
        if vectors.shape[1] != EMBEDDING_DIM:
            raise ValueError(f'Embedding dimension mismatch: {vectors.shape[1]} vs {EMBEDDING_DIM}')
        for start in range(0, len(chunks), 64):
            points = [PointStruct(id=i, vector=vectors[i].tolist(),
                                  payload={'text': chunks[i]['text'], 'metadata': dict(chunks[i].get('metadata', {}))})
                      for i in range(start, min(start+64, len(chunks)))]
            self.client.upsert(collection, points, wait=True)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0 or not query.strip() or not self.client.collection_exists(collection):
            return []
        vector = self._query_vectors.get(query)
        if vector is None:
            vector = self._get_encoder().encode(query, normalize_embeddings=True).tolist()
        response = self.client.query_points(collection, query=vector, limit=top_k, with_payload=True)
        return [SearchResult(pt.payload['text'], float(pt.score), dict(pt.payload.get('metadata', {})), 'dense')
                for pt in response.points]

    def prepare_queries(self, queries):
        unique = list(dict.fromkeys(queries))
        vectors = self._get_encoder().encode(unique, normalize_embeddings=True, batch_size=4)
        self._query_vectors.update({q: v.tolist() for q, v in zip(unique, vectors)})
        self._encoder = None
        from src.model_cache import release_sentence_models
        release_sentence_models()


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError('k must be non-negative')
    merged = {}
    for results in results_list:
        seen = set()
        for rank, result in enumerate(results):
            identity = result.metadata.get('chunk_id') or (result.metadata.get('source'), result.text)
            if identity in seen:
                continue
            seen.add(identity)
            if identity not in merged:
                merged[identity] = [0.0, result]
            merged[identity][0] += 1.0/(k+rank+1)
    ordered = sorted(merged.values(), key=lambda entry: -entry[0])[:max(0, top_k)]
    return [SearchResult(r.text, score, dict(r.metadata), 'hybrid') for score, r in ordered]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print("Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
