from __future__ import annotations

import re
from collections import defaultdict

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.document import DocumentChunk
from app.models.knowledge_base import KnowledgeBase
from app.services.embedding_service import embed_query
from app.vectorstore.chroma_manager import search_similar_chunks


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text.lower())
    return [tok for tok in cleaned.split() if tok]


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s:
        return {key: 1.0 for key in scores}
    return {key: (val - min_s) / (max_s - min_s) for key, val in scores.items()}


def _fuse_weighted(
    bm25_scores: dict[str, float], dense_scores: dict[str, float], alpha: float
) -> dict[str, float]:
    bm25_norm = _normalize_scores(bm25_scores)
    dense_norm = _normalize_scores(dense_scores)
    keys = set(bm25_norm.keys()) | set(dense_norm.keys())
    return {
        key: alpha * bm25_norm.get(key, 0.0) + (1.0 - alpha) * dense_norm.get(key, 0.0)
        for key in keys
    }


def _fuse_rrf(
    bm25_rank: list[str], dense_rank: list[str], k: int = 60
) -> dict[str, float]:
    score_map: dict[str, float] = defaultdict(float)
    for rank, chunk_id in enumerate(bm25_rank):
        score_map[chunk_id] += 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(dense_rank):
        score_map[chunk_id] += 1.0 / (k + rank + 1)
    return dict(score_map)


def hybrid_retrieve(
    db: Session,
    query: str,
    kb_ids: list[str],
    top_k: int,
    score_threshold: float,
    fusion_mode: str,
    alpha: float,
) -> list[dict]:
    chunk_rows = list(
        db.scalars(select(DocumentChunk).where(DocumentChunk.kb_id.in_(kb_ids))).all()
    )
    if not chunk_rows:
        return []

    corpus = [row.content for row in chunk_rows]
    corpus_tokens = [_tokenize(text) for text in corpus]
    query_tokens = _tokenize(query)
    bm25 = BM25Okapi(corpus_tokens)
    bm25_raw_scores = bm25.get_scores(query_tokens)

    bm25_scores: dict[str, float] = {}
    bm25_ranked: list[str] = []
    for idx, row in enumerate(chunk_rows):
        score = float(bm25_raw_scores[idx])
        bm25_scores[row.id] = score
    bm25_ranked = [
        cid
        for cid, _ in sorted(
            bm25_scores.items(), key=lambda item: item[1], reverse=True
        )[: top_k * 2]
    ]

    dense_scores: dict[str, float] = {}
    dense_ranked: list[str] = []
    chunk_cache = {row.id: row for row in chunk_rows}
    kb_model_map = {
        item.id: item.embedding_model
        for item in db.scalars(
            select(KnowledgeBase).where(KnowledgeBase.id.in_(kb_ids))
        ).all()
    }
    for kb_id in kb_ids:
        model_name = kb_model_map.get(kb_id, "bge-small-zh-v1.5")
        query_embedding = embed_query(query=query, model_name=model_name)
        for item in search_similar_chunks(
            kb_id=kb_id,
            query=query,
            top_k=top_k * 2,
            query_embedding=query_embedding,
        ):
            chunk_id = item["chunk_id"]
            if chunk_id in chunk_cache:
                dense_scores[chunk_id] = max(
                    dense_scores.get(chunk_id, 0.0), item["score"]
                )
    dense_ranked = [
        cid
        for cid, _ in sorted(
            dense_scores.items(), key=lambda item: item[1], reverse=True
        )[: top_k * 2]
    ]

    if fusion_mode == "rrf":
        fused = _fuse_rrf(bm25_ranked, dense_ranked)
    else:
        fused = _fuse_weighted(bm25_scores, dense_scores, alpha=alpha)

    sorted_ids = [
        chunk_id
        for chunk_id, score in sorted(
            fused.items(), key=lambda item: item[1], reverse=True
        )
        if score >= score_threshold
    ][:top_k]

    results: list[dict] = []
    for chunk_id in sorted_ids:
        row = chunk_cache[chunk_id]
        results.append(
            {
                "chunk_id": row.id,
                "document_id": row.document_id,
                "kb_id": row.kb_id,
                "content": row.content,
                "source_location": row.source_location,
                "score": float(fused.get(chunk_id, 0.0)),
            }
        )
    return results
